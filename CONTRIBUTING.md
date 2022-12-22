# Contributing

pace is actively developed by AI2, so please contact us if there is interest in making contributions in the near-term.
Contributors names will be added to [`CONTRIBUTORS.md`](https://github.com/VulcanClimateModeling/fv3core/blob/master/CONTRIBUTORS.md).

## Linting

Dependencies for linting are maintained in `requirements_lint.txt`, and can be installed with:

```shell
$ pip install -c constraints.txt -r requirements_lint.txt
```

Correcting and checking your code complies with all requirements can be run with:

```shell
$ make lint
```

We manage the list of syntax requirements using [pre-commit](https://pre-commit.com/).
**This runs all checks and is required to pass as one of the continuous integration tests.**

The list of checkes includes `black`, `isort`, and `flake8`, among a few others, found in `.pre-commit-config.yaml`.
[Black](https://github.com/ambv/black) is configured in `pyproject.toml`, and the others use `setup.cfg`.

We mostly use standard Flake8, but ignore the following rules:

- `W503: line break before binary operator`
    - We should choose whether to ignore W503 or W504, and only ignore one
- `E203: whitespace before ':'`
    - Needs to be ignored to be consistent with black
- `E302: Expected 2 blank lines, found 0`
- `F841: local variable is assigned to but never used`
    - Must ignore because gt4py stencils return outputs in place (no return statement)
    - Can avoid if all stencil assignments use square brackets on the left
        - `out[0, 0, 0]` or `out[idx]/out[curr]/out[something]`

Flake8 rules not listed will be enforced unless we find a need not to enforce them.
Since documentation does go out of date, please consult the entry in `setup.cfg` for the most up-to-date requirements and update this document if you notice a difference.

## Style

The first version of the dycore was written with minimal metadata and typing, motivated primarily by matching the regression data produced by the Fortran version of the code using the numpy backend.
We are now actively refactoring, moving code that still does computations in Python into GT4py stencils and merging stencils together with the introduction of enabling features in GT4py (such as regions).
While we do that, clarifying the operation of the model and what the variables are will both help make the model easier to read and reduce errors as we move around long lists of argument variables.

First, checking you are following the guidelines established at [Code Review Checklist](https://paper.dropb\
ox.com/doc/Code-Review-Checklist--BD7zigBMAhMZAPkeNENeuU2UAg-IlsYffZgTwyKEylty7NhY) when writing new code.

For code visited in refactors, we want to start adding the following where appropriate:
- Type hints on Python functions (see [`fv3core/utils/typing.py`](https://github.com/VulcanClimateModeling/fv3core/blob/master/fv3core/utils/typing.py) and below)
- More descriptive types on stencil definitions
- Docstrings on outward facing Python functions: describe what methods are doing, describe the intent (*in*, *out*, or *inout*) of the function arguments

### Docstrings
These should aid us in refactoring and understanding what a function is doing. If it is not completely understood yet what is happening, please consult other team members and GFDL as appropriate. If we still cannot tell what is happening, you can write what is known along with a `TODOC` to indicate it is incomplete.
For example:

```python
def stencil(...):
    """This is a short description that fits on one line.

    Here is a longer explanation of the assumptions and why this exists.

    TODOC: Why does ke_c need to be updated here?

    Args:
        uc: x-velocity on C-grid (inout)
        vc: y-velocity on C-grid (inout)
        vort_c: Vorticity on C-grid (inout)
        ke_c: kinetic energy on C-grid (inout)
        v: y-velocity on D-grid (inout)
        u: x-velocity on D-grid (inout)
        dt2: timestep (in)

    """
```


### Type hinting Python functions
These should mostly be lightweight workflow wrappers calling gt4py stencils, though currently exceptions exist where Python code does computations on data fields.

New code should be type hinted making use of `fv3core/utils/typing.py` when typing gt4py fields. You may run into code in fv3core before we added this convention. An older code like:
```python
def compute(var1, var2, var3, param1, param2, param3):
```

would become:

```python
def compute(var1: FloatField, var2:IntField, var3: BoolField,
            param1: float_type, param2: int_type, param3: bool_type):
```
There is no determined convention for order of arguments, but the code generally follows the convention of listing 3d fields first followed by parameters, as is required by gt4py stencil functions.

Another example
```python
def make_storage_from_shape(shape, origin, dtype):
```

Turns into
```python
    def make_storage_from_shape(
        shape: Tuple[int, int, int],
        origin: Tuple[int, int, int],
        *,
        dtype: DTypes = np.float64,
    ) -> Field:
```

- We prefer to add typing to methods that are used by other modules.
  Not every internal method needs this level of specification, in particular module-private routines or routines which are already self-descriptive without type hinting.
- Internal functions that are likely to be inlined into a larger stencil do not need this if it will just be removed in the near-term.

### GT4Py stencils
We interface to `gt4py.cartesian.gtscript.stencil` through pace.dsl.stencil, specifically the FrozenStencil, that allows us to minimize runtime overhead in calling stencils.


```python
@gtstencil
def pt_adjust(pkz:FloatField, dp1: FloatField, q_con: FloatField, pt: FloatField):
    with computation(PARALLEL), interval(...):
```

[`fv3core/utils/typing.py`](https://github.com/VulcanClimateModeling/fv3core/blob/master/fv3core/utils/typing.py) defines various field types.
For example, `FloatField[IJ]` for a 2D field of default floating point values.


### GTScript functions
These use the `@gtscript.function` decorator and the arguments do not include type
specifications. They will continue to not have type hinting.

e.g.:

    @gtscript.function
    def get_bl(al, q):

### Assertions
We can now include assertions of compile time variables inside of gtscript functions with the syntax `compile_assert(<expression>)`, for example `compile_assert(namelist.grid_type < 3)`.

### State
Some outer functions include a 'state' object that is a SimpleNamespace of variables and a `comm` object that is the `CubedSphereCommunicator` object enabling halo updates.
The `state` include pointers to gt4py storages for all variables used in the method.
For fields that experience a halo update, the state includes pointers to Quantity objects named `<storage variable name>_quantity`, which is a lightweight wrapper around the storage.
This enables using gt4py storages in stencils and quantities for halo updates, using the same memory space.
A future refactor will simplify this convention, likely through the use of the decorator and/or a GDP from GT4py that may allow Quantities to be used in stencils.

As we refactor, we may opt to use this convention more (or a similar one to avoid calling functions while relying on getting the order of a long list of variables correct), but should be considered as part of a refactor on a case-by-case basis.


### New styles
Propose new style ideas in a meeting or github issue to the team (or subset) with examples and description of how data flow would be altered if relevant. Once an idea is accepted, open a PR with the idea applied to a sample if possible (if not, correct the whole model), and update this doc to reflect the new convention we all should incorporate as we refactor.
Share news of this update when the PR is accepted and merged, including guidelines for using the new convention.
Implementers and reviewers of new code changes should consider whether the new style should be applied at the same time so we can introduce this change in a piecemeal fashion rather than disrupting every active task.

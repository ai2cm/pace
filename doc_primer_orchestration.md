DaCe Orchestration in Pace: a primer
====================================


Fundamentals
------------

Full program optimziation with DaCe is the process of turning all Python and GT4Py code in generated C++.

_Orchestration_ is our own wording for full program optimization. We only _orchestrate_ the runtime code of the model, e.g. everything in the `__call__` method of the module. All code in `__init__` is executed like a normal gt backend.

At the highest level in Pace, to turn on orchestration you need to flip the `FV3_DACEMODE` to an orchestrated options _and_ run a `dace:*` backend (it will error out if run anything else). Option for `FV3_DACEMODE` are:
- _Python_: default, turns orchestration off.
- _Build_: build the SDFG then exit without running. See Build for limitation of build strategy.
- _BuildAndRun_: as above, but distribute the build and run.
- _Run_: tries to execute, errors out if the cache don't exists.

Code is orchestrated two ways:
- functions are orchestrated via `orchestrate_function` decorator,
- methods are orchestrate via the `orchestrate` function (e.g. `pace.driver.Driver._critical_path_step_all`)

The later is the way we orchestrate in our model. `orchestrate` is often called as the first function in the `__init__`. It patches _in place_ the methods and replace them with a wrapper that will deal with turning it all into executable SDFG when call time comes.

The orchestration has two parameters: config (will expand later) and `dace_compiletime_args`.

DaCe needs to be described all memory so it can interface it in the C code that will be executed. Some memory is automatically parsed (e.g. numpy, cupy, scalars) and others need description. In our case `Quantity` and others need to be flag as `dace.compiletime` which tells DaCe to not try to AOT the memory and wait for JIT time. The `dace_compiletime_args` helps with tagging those without having to change the type hint.

File structure
--------------

`pace.dsl.dace.*` carries the structure for orchestration.
* `build.py`: tooling for distributed build & SDFG load.
* `dace_config.py`: DaCeConfig & DaCeOrchestration enum.
* `orchestration.py`: main code, takes care of orchestration .scaffolding, build pipeline (including parsing) and execution.
* `sdfg_opt_passes.py`: custom optimization pass for Pace, used in the build pipeline.
* `utils.py`: as every "utils" or "misc" or "common" file, this should not exists and collect tools & functions I lazily didn't put in a proper place.
* `wrapped_halo_exchange.py`: a callback-ready halo exchanger, which is our current solution for keeping the Halo Exchange in python (because of prior optimization) in orchestration.

DaCe Config
-----------

DaCe has many configuration options. When executing, it drops or reads a `dace.conf` to get/set options for execution. Because this is a performance-portable model and not a DaCe model, decision has been taken to freeze the options.

`pace.dsl.dace.dace_config` carries a set of tested options for DaCe, with doc. It also takes care of removing the `dace.conf` that will be generated automatically when using DaCe. Documentation should be self-explanatory but a good one to remember is:

```python
# Enable to debug GPU failures
dace.config.Config.set("compiler", "cuda", "syncdebug", value=False)
```
When set to `True`, this will drop a few checks:
- `sdfg_nan_checker`, which drops a NaN check after _every_ computation on field _written_.
- `negative_qtracers_checker` drops a check for `tracer < -1e8` for every written field named one of the tracers
- `negative_delp_checker` drops a check for `delp < -1e8` for every written field named `delp*`
See `dsl/pace/dsl/dace/utils.py` for details.

Build
-----

Orchestrated code won't build the same way the gt backend builds. The build pipeline will lead to a single folder with code & `.so`. In the case of the driver main call, this would be in `.gt_cache_*/dacecache/pace_driver_driver_Driver__critical_path_step_all`.

Code goes through phases before being ready to execute:
* stencils are `parsed` into non-expanded SDFG (gt4py takes care of this),
* all code is `parsed` into a single SDFG with stencils' SDFG included (dace takes care of this and the following steps),
* a first `simplify` is applied to the SDFG to optimize the memory flow,
* we apply the custom `splittable_region_expansion` which optimize small regions (_major_ speed up),
* `expand` will expand all the stencils to a fully workable SDFG (with tasklet filled)
* another `simplify` is applied,
* the memory that can is flagged to be `pooled`,
* [OPTIONAL] Insert debugging passes
* `code generation` into a single file for CPU or two for GPU (a `.cpp` and a `.cu`),
* the SDFG is analysed for memory consumption.


Orchestration comes with it's own distributed compilation (could be merged with gt). It compiles the top tile and distriubutes the results to other ranks. This uses a couple of hypothesis that limits how to build/execute. The major one is that any decomposition from `(3,3)` upward will require the following workflow:
- compile on `(3,3)`,
- copy caches 0 to 8 (top tile) to target decomposition run dir,
- execute (`FV3_DACEMODE=Run`) target decompoposition.

The other limitations is that it only is protecting bad runs when resolution X == resolution Y.

The distributed compilation can be deactivated in `dace_config.py` by turning `DEACTIVATE_DISTRIBUTED_DACE_COMPILE` to `True`.

Execution
---------

If orchestration gives us the best speed, remember that it execute _one single .so_ and therefore cannot be interrupted with regular Python debugging techniques.

Pitfalls & remedies
-------------------

_Callback_

Escape DaCe by decorating a function with `@dace_inibitor`, this will bounce out of the C back into Python. It will carry `self` over and any "simple" arguments. Can return scalars.

_Scalar iniling_

DaCe will optimize aas much as it can. This means any scalar with be turned into the value if they are never written to. This can lead to subtle bug because of our mode of distributed build. For example `da_min` is a grid variable which is decomposition dependant and read-only in the critical code path. This will be inlined, leading to error when executing larger decomposition (because it uses the `da_min` of the build decomposition). Our workaround for now is a callback (pending a series of fix from DaCe for a better solution).

_Parsing errors_

DaCe cannot parse _any_ dynamic Python and any code that allocates memory on the fly (think list creation). It will also complain about any arguments it can't memory describe (remember `dace_compiletime_args` ).

Conclusion
----------

It is not halo exchange.

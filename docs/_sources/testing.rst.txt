=======
Testing
=======

Savepoint tests are run automatically on every commit to the main branch.
Savepoint data are generated from `fv3gfs-fortran`_ and can also be downloaded:

.. code-block:: console

    $ make get_test_data
    $ # if you do not have access to the Google Cloud Storage bucket, use FTP:
    $ make USE_FTP=yes get_test_data

Savepoint data are used in the "translate" tests and in checkpointer tests.
Developers should be aware that the "translate" tests are an older, initial design of the test infrastructure which has grown organically and may be difficult to understand or modify, but currently covers smaller parts of the code not tested independently by the checkpointer tests.
In the long run we suggest increasing the number of checkpoints and adding new checkpointer tests, and eventually removing the translate tests, which are considered deprecated.

#. Individual translate tests

    These test at the module level such as `c_sw` and `d_sw`, and the translate logic is shared among dynamical core and physics.
    Larger tests also exist such as `translate_fvdynamics` which tests a full acoustic time step.
    Manual thresholds are set for each savepoint test. Curerntly, maximum threshold is applied to all variables within the test.
    Additionally, a near-zero value can be specified for a variable to ignore values that are very close to zero.

#. Checkpointer tests

    This tests the full model run where checkpoints are inserted throughout the model.
    See ``tests/savepoint/test_checkpoints.py`` for an example.
    Checkpointers are given model state along with a label, and may implement any behavior they wish.
    For example, checkpointers have been written to:

    #. compare the model state to a reference state (:py:class:`pace.util.ValidationCheckpointer`)
    #. calibrate the threshold for each variable given a perturbed state (:py:class:`pace.util.ThresholdCalibrationCheckpointer`)

    Additional checkpoint behaviors could be implemented, for example to save reference test data directly from Python.
    Thresholds are set automatically using a :py:class:`pace.util.ThresholdCalibrationCheckpointer` for each variable based on a round-off error perturbed initial state.
    We run the model multiple times with a perturbed initial state and record the largest differences at each checkpoint for each variable.
    The threshold is then set to the largest difference multiplied by a scaling factor.
    Currently, only checkpoint tests within the dynamical core are tested.
    There are two outstanding PRs to include driver and physics checkpoint tests.

-----------
Limitations
-----------
While individual translate test can be run on all backends, checkpointer tests do not work for orchestrated DaCe backend.
This is a limitation due to DaCe not accepting keyword arguments or a list of :py:class:`pace.util.Quantity`, causing the checkpointer calls to be overly complicated.
A possible workaround is to follow the HaloUpdater example to wrap the variables at init time and called during DaCe callbacks.
A better solution would be to have DaCe accept a list of :py:class:`pace.util.Quantity`.

--------
Examples
--------
Translate tests for the dynamical core can be run as follows:

.. code-block:: console

    $ make savepoint_tests

We suggest reading the Makefile for a full list of translate test targets. Checkpointer tests can be run as follows:

.. code-block:: console

    $ make test_savepoint

.. _`fv3gfs-fortran`: https://github.com/ai2cm/fv3gfs-fortran/tree/master/tests/serialized_test_data_generation

=======
Testing
=======

Savepoint tests are run automatically on every commit to the main branch.
Savepoint data are generated from `fv3gfs-fortran`_ and can also be downloaded:

.. code-block:: console

    $ make get_test_data
    $ # if you do not have access to the Google Cloud Storage bucket, use FTP:
    $ make USE_FTP=yes get_test_data

Savepoint data are used in two ways:

#. Individual translate tests

    This tests at the module level such as `c_sw` and `d_sw`, and the translate logic is shared among dynamical core and physics.
    Larger tests also exist such as `translate_fvdynamics` which tests a full acoustic time step.
    Manual thresholds are set for each savepoint test. Curerntly, maximum threshold is applied to all variables within the test.
    Additionally, a near-zero value can be specified for a variable to ignore values that are very close to zero.

#. Checkpointer tests

    This tests the full model run where checkpoints are inserted throughout the model.
    During these checkpoints, it is possible to

    #. compare the model state to a reference state
    #. calibrate the threshold for each variable given a perturbed state

    Additional checkpoint behaviors can be implemented.
    Thresholds are set automatically for each variable based on a round-off error perturbed initial state.
    E.g., we run the model multiple times with a perturbed initial state and record the largest differences at each checkpoint for each variable.
    The threshold is then set to the largest difference plus a small tolerance.
    Currently, only checkpoint tests within the dynamical core are tested.
    There are two outstanding PRs to include driver and physics checkpoint tests.

While individual translate test can be run on all backends, checkpointer tests do not work for orchestrated DaCe backend.

Translate tests can be run as follows:

.. code-block:: console

    $ make savepoint_tests

Checkpointer tests can be run as follows:

.. code-block:: console

    $ make test_savepoint

.. _`fv3gfs-fortran`: https://github.com/ai2cm/fv3gfs-fortran/tree/master/tests/serialized_test_data_generation

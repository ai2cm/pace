import gt4py
import numpy as np

dict_dimension = {
    "phii": 80,
    "prsi": 80,
    "phil": 79,
    "pt": 79,
    "qvapor": 79,
    "del": 79,
    "del_gz": 79,
    "w": 79,
    "dz": 79,
    "p123": 79,
    "omga": 79,
    "qv1": 79,
    "gt0": 79,
    "prsl": 79,
    "delp": 79,
    "graupel0": 79,
    "pt": 79,
    "pt_dt": 79,
    "qv_dt": 79,
    "ql_dt": 79,
    "qr_dt": 79,
    "qi_dt": 79,
    "qs_dt": 79,
    "qg_dt": 79,
    "qa_dt": 79,
    "udt": 79,
    "vdt": 79,
    "qa1": 79,
    "qg1": 79,
    "qi1": 79,
    "ql1": 79,
    "qr1": 79,
    "qs1": 79,
    "qv1": 79,
    "uin": 79,
    "vin": 79,
}

for rank in range(6):
    fv3gfs_physics = np.load(
        "integrated_after_physics_driver_rank_" + str(rank) + ".npy", allow_pickle=True
    ).item()
    standalone = np.load(
        "standalone_after_physics_driver_rank_" + str(rank) + ".npy", allow_pickle=True
    ).item()
    for key in fv3gfs_physics.keys():
        if key in standalone.keys():  # if variable is in both versions
            if isinstance(fv3gfs_physics[key], float):
                var1 = standalone[key]
                var2 = fv3gfs_physics[key]
            elif len(fv3gfs_physics[key].shape) < 1:
                var1 = standalone[key].data
                var2 = fv3gfs_physics[key]
            elif len(fv3gfs_physics[key].shape) == 2:
                var1 = standalone[key].data[:, :, 0]
                var2 = np.reshape(
                    fv3gfs_physics[key].data[3:-4, 3:-4], (144, 1), order="F"
                )
            else:
                if key in dict_dimension and dict_dimension[key] == 80:
                    if isinstance(standalone[key], np.ndarray):
                        var1 = standalone[key]
                    else:
                        var1 = standalone[key].data
                    var2 = fv3gfs_physics[key].data
                    var2 = np.reshape(
                        var2[3:-4, 3:-4, :], (144, 1, dict_dimension[key]), order="F"
                    )

                else:
                    if isinstance(standalone[key], np.ndarray):
                        var1 = standalone[key][:, :, 1:]
                    else:
                        var1 = standalone[key].data[:, :, 1:]
                    var2 = fv3gfs_physics[key].data[:, :, 0:-1]
                    var2 = np.reshape(var2[3:-4, 3:-4, :], (144, 1, 79), order="F")

            print(rank, key)
            try:
                # print(key, "passed ", var1.max())
                np.testing.assert_allclose(var1, var2, atol=1e-16)
            except:
                print(key, " did not pass")
                print(var1[0, 0, :], var2[0, 0, :])
                continue
            ind = np.array(np.nonzero(~np.isclose(var1, var2, equal_nan=True)))

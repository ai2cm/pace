import numpy as np

dict_dimension = {
    "phii": 80,
    "prsi": 80,
    "phil": 79,
    "pt": 79,
    "qvapor": 79,
    "del": 79,
    "del_gz": 79,
}

fv3gfs_physics = np.load("integrated_after_phifv3_rank0.npy", allow_pickle=True).item()
standalone = np.load("standalone_after_phifv3_rank_0.npy", allow_pickle=True).item()
for key in standalone.keys():
    if dict_dimension[key] == 79:
        var1 = standalone[key].data[:, :, 1:]
        var2 = fv3gfs_physics[key].data[:, :, 0:-1]
    else:
        var1 = standalone[key].data
        var2 = fv3gfs_physics[key].data
    var2 = np.reshape(var2[3:-4, 3:-4, :], (144, 1, dict_dimension[key]), order="F")

    print(key)
    try:
        np.testing.assert_allclose(var1, var2)
    except:
        print(key, " did not pass")
        print(var1[0, 0, :], var2[0, 0, :])
        continue
    ind = np.array(np.nonzero(~np.isclose(var1, var2, equal_nan=True)))

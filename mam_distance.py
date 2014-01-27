import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from dipy.tracking.distances import mam_distances, bundles_distances_mam
from time import time
from sklearn.metrics import pairwise_distances
# from scipy.spatial import KDTree
# from scipy.spatial import cKDTree as KDTree
from sklearn.neighbors import KDTree


def avg_mam_distance_dipy(s1, s2):
    """Zhang (2008) streamline distance provided by DiPy.
    """
    return mam_distances(s1, s2, metric='avg')


def avg_mam_distance_numpy_scipy(s1, s2):
    """Zhang (2008) streamline distance using NumPy broadcasting and
    SciPy distance_matrix().
    """
    dm = cdist(s1, s2)
    return 0.5 * (dm.min(0).mean() + dm.min(1).mean())


def avg_mam_distance_numpy(s1, s2):
    """Zhang (2008) streamline distance using just NumPy broadcasting.
    """
    dm = np.sqrt((s1 * s1).sum(1)[:, None] - 2.0 * np.dot(s1, s2.T) + (s2 * s2).sum(1))
    return 0.5 * (dm.min(0).mean() + dm.min(1).mean())


def avg_mam_distance_numpy_faster(s1, s2):
    """Zhang (2008) streamline distance using just NumPy broadcasting
    and a simple trick to reduce the number of sqrt() evaluations
    (which impacts on long streamlines).
    """
    dm = (s1 * s1).sum(1)[:, None] - 2.0 * np.dot(s1, s2.T) + (s2 * s2).sum(1)
    return 0.5 * (np.sqrt(dm.min(0)).mean() + np.sqrt(dm.min(1)).mean())


def avg_mam_distance_numpy_faster2(s1, s2):
    """Zhang (2008) streamline distance using just NumPy broadcasting
    and a simple trick to reduce the number of sqrt() evaluations
    (which impacts on long streamlines).
    """
    dmx = np.subtract.outer(s1[:,0], s2[:,0])
    dmy = np.subtract.outer(s1[:,1], s2[:,1])
    dmz = np.subtract.outer(s1[:,2], s2[:,2])
    dm = dmx * dmx + dmy * dmy + dmz * dmz
    return 0.5 * (np.sqrt(dm.min(0)).mean() + np.sqrt(dm.min(1)).mean())


def avg_mam_distance_numpy_faster3(s1, s2):
    """Zhang (2008) streamline distance using just NumPy broadcasting
    and a simple trick to reduce the number of sqrt() evaluations
    (which impacts on long streamlines).
    """
    dm = pairwise_distances(s1, s2, metric='sqeuclidean')
    return 0.5 * (np.sqrt(dm.min(0)).mean() + np.sqrt(dm.min(1)).mean())


def avg_mam_distance_numpy_faster4(s1, s2):
    """Zhang (2008) streamline distance using just NumPy broadcasting
    and a simple trick to reduce the number of sqrt() evaluations
    (which impacts on long streamlines).
    """
    dmx = s1[:,0][:, None] - s2[:,0]
    dmy = s1[:,1][:, None] - s2[:,1]
    dmz = s1[:,2][:, None] - s2[:,2]
    dm = dmx * dmx + dmy * dmy + dmz * dmz
    return 0.5 * (np.sqrt(dm.min(0)).mean() + np.sqrt(dm.min(1)).mean())


def avg_mam_distance_numpy_faster5(s1, s2):
    """Zhang (2008) streamline distance using KDTree.
    """
    kdt1 = KDTree(s1)
    kdt2 = KDTree(s2)
    return 0.5 * (kdt2.query(s1, k=1)[0].mean() + kdt1.query(s2, k=1)[0].mean())


def avg_mam_distance_numpy_faster6(s1, s2, kdt1, kdt2):
    """Zhang (2008) streamline distance using KDTree.query() with
    precomputed trees.
    """
    return 0.5 * (kdt2.query(s1, k=1)[0].mean() + kdt1.query(s2, k=1)[0].mean())


def avg_mam_distance_numpy_faster7(s1, s2):
    """Zhang (2008) streamline distance using just NumPy broadcasting
    and a simple trick to reduce the number of sqrt() evaluations
    (which impacts on long streamlines).
    """
    dm = s1[:, None, :] - s2[None, :, :]
    dm = (dm * dm).sum(2)
    return 0.5 * (np.sqrt(dm.min(0)).mean() + np.sqrt(dm.min(1)).mean())


if __name__ == '__main__':

    np.random.seed(0)

    s1 = np.random.random((100,3))
    s2 = np.random.random((200,3))

    print "avg_mam_distance_dipy(s1, s2) =", avg_mam_distance_dipy(s1, s2)

    print "avg_mam_distance_numpy_scipy(s1, s2) =", avg_mam_distance_numpy_scipy(s1, s2)

    print "avg_mam_distance_numpy(s1, s2) =", avg_mam_distance_numpy(s1, s2)

    print "avg_mam_distance_numpy_faster(s1, s2) =", avg_mam_distance_numpy_faster(s1, s2)

    print "avg_mam_distance_numpy_faster2(s1, s2) =", avg_mam_distance_numpy_faster2(s1, s2)

    print "avg_mam_distance_numpy_faster3(s1, s2) =", avg_mam_distance_numpy_faster3(s1, s2)

    print "avg_mam_distance_numpy_faster4(s1, s2) =", avg_mam_distance_numpy_faster4(s1, s2)

    print "avg_mam_distance_numpy_faster5(s1, s2) =", avg_mam_distance_numpy_faster5(s1, s2)

    kdt1 = KDTree(s1)
    kdt2 = KDTree(s2)
    print "avg_mam_distance_numpy_faster6(s1, s2) =", avg_mam_distance_numpy_faster6(s1, s2, kdt1, kdt2)

    print

    print "Time to compute the distance between two bundles:"

    B1_size = 100
    B2_size = 200
    streamline_max_size = 200
    streamline_min_size = 10

    print "B1 size =", B1_size
    print "B2 size =", B2_size
    print "streamline_min_size =", streamline_min_size
    print "streamline_max_size =", streamline_max_size

    B1 = []
    for i in range(B1_size):
        n1 = np.random.randint(streamline_min_size, streamline_max_size)
        B1.append(np.random.random((n1, 3)))

    B2 = []
    for i in range(B2_size):
        n2 = np.random.randint(streamline_min_size, streamline_max_size)
        B2.append(np.random.random((n2, 3)))

    print "DiPy's bundles_distances_mam():",
    t0 = time()
    mam_dm_dipy = bundles_distances_mam(B1, B2, metric='avg')
    print time() - t0 , 'sec.'

    print "Python loop + avg_mam_distance_dipy:", 
    t0 = time()
    mam_dm_python_dipy = np.zeros((len(B1), len(B2)))
    for i in range(len(B1)):
        for j in range(len(B2)):
            mam_dm_python_dipy[i, j] = avg_mam_distance_dipy(B1[i], B2[j])

    print time() - t0 , 'sec.'
    np.testing.assert_almost_equal(mam_dm_dipy, mam_dm_python_dipy, decimal=5)
                         
    print "Python loop + NumPy + Scipy:", 
    t0 = time()
    mam_dm_python_numpy_scipy = np.zeros((len(B1), len(B2)))
    for i in range(len(B1)):
        for j in range(len(B2)):
            mam_dm_python_numpy_scipy[i, j] = avg_mam_distance_numpy_scipy(B1[i], B2[j])

    print time() - t0 , 'sec.'
    np.testing.assert_almost_equal(mam_dm_python_dipy, mam_dm_python_numpy_scipy, decimal=5)

    print "Python loop + NumPy:",
    t0 = time()
    mam_dm_python_numpy = np.zeros((len(B1), len(B2)))
    for i in range(len(B1)):
        for j in range(len(B2)):
            mam_dm_python_numpy[i, j] = avg_mam_distance_numpy(B1[i], B2[j])

    print time() - t0 , 'sec.'
    np.testing.assert_almost_equal(mam_dm_python_numpy_scipy, mam_dm_python_numpy, decimal=5)

    print "Python loop + NumPy (faster implementation):", 
    t0 = time()
    mam_dm_python_numpy_faster = np.zeros((len(B1), len(B2)))
    for i in range(len(B1)):
        for j in range(len(B2)):
            mam_dm_python_numpy_faster[i, j] = avg_mam_distance_numpy_faster(B1[i], B2[j])

    print time() - t0 , 'sec.'
    np.testing.assert_almost_equal(mam_dm_python_numpy, mam_dm_python_numpy_faster, decimal=5)

    print "Python loop + NumPy (faster2 implementation):", 
    t0 = time()
    mam_dm_python_numpy_faster2 = np.zeros((len(B1), len(B2)))
    for i in range(len(B1)):
        for j in range(len(B2)):
            mam_dm_python_numpy_faster2[i, j] = avg_mam_distance_numpy_faster2(B1[i], B2[j])

    print time() - t0 , 'sec.'
    np.testing.assert_almost_equal(mam_dm_python_numpy_faster, mam_dm_python_numpy_faster2, decimal=5)

    print "Python loop + sklearn's pairwise_distances():", 
    t0 = time()
    mam_dm_python_numpy_faster3 = np.zeros((len(B1), len(B2)))
    for i in range(len(B1)):
        for j in range(len(B2)):
            mam_dm_python_numpy_faster3[i, j] = avg_mam_distance_numpy_faster3(B1[i], B2[j])

    print time() - t0 , 'sec.'
    np.testing.assert_almost_equal(mam_dm_python_numpy_faster2, mam_dm_python_numpy_faster3, decimal=5)

    print "Python loop + NumPy (faster4 implementation):", 
    t0 = time()
    mam_dm_python_numpy_faster4 = np.zeros((len(B1), len(B2)))
    for i in range(len(B1)):
        for j in range(len(B2)):
            mam_dm_python_numpy_faster4[i, j] = avg_mam_distance_numpy_faster4(B1[i], B2[j])

    print time() - t0 , 'sec.'
    np.testing.assert_almost_equal(mam_dm_python_numpy_faster3, mam_dm_python_numpy_faster4, decimal=5)


    print "Python loop + KDTree (faster5 implementation):", 
    t0 = time()
    mam_dm_python_numpy_faster5 = np.zeros((len(B1), len(B2)))
    for i in range(len(B1)):
        for j in range(len(B2)):
            mam_dm_python_numpy_faster5[i, j] = avg_mam_distance_numpy_faster5(B1[i], B2[j])

    print time() - t0 , 'sec.'
    np.testing.assert_almost_equal(mam_dm_python_numpy_faster4, mam_dm_python_numpy_faster5, decimal=5)


    print "Python loop + KDTree (faster6 implementation):", 
    t0 = time()
    mam_dm_python_numpy_faster6 = np.zeros((len(B1), len(B2)))
    kdtB2 = [KDTree(s2) for s2 in B2]
    for i in range(len(B1)):
        kdt1 = KDTree(B1[i])
        for j in range(len(B2)):
            mam_dm_python_numpy_faster6[i, j] = avg_mam_distance_numpy_faster6(B1[i], B2[j], kdt1, kdtB2[j])

    print time() - t0 , 'sec.'
    np.testing.assert_almost_equal(mam_dm_python_numpy_faster5, mam_dm_python_numpy_faster6, decimal=5)

    
    print "Python loop + NumPy (faster7 implementation):", 
    t0 = time()
    mam_dm_python_numpy_faster7 = np.zeros((len(B1), len(B2)))
    for i in range(len(B1)):
        for j in range(len(B2)):
            mam_dm_python_numpy_faster7[i, j] = avg_mam_distance_numpy_faster7(B1[i], B2[j])

    print time() - t0 , 'sec.'
    np.testing.assert_almost_equal(mam_dm_python_numpy_faster6, mam_dm_python_numpy_faster7, decimal=5)

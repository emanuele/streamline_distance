"""Speed comparison of different implementations of the mean average
minimum distance between two streamlines (and two bundles) as descibed
in Zhang (2008).
"""

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
    """Average streamline distance provided by DiPy (Zhang (2008))
    """
    return mam_distances(s1, s2, metric='avg')


def avg_mam_distance_scipy(s1, s2):
    """Streamline distance using SciPy cidst()

    Note: after a recent patch I submitted to SciPy, the speed of this
    implementation is greatly improved.
    """
    dm = cdist(s1, s2, metric='sqeuclidean')
    return 0.5 * (np.sqrt(dm.min(0)).mean() + np.sqrt(dm.min(1)).mean())


def avg_mam_distance_numpy(s1, s2):
    """Just NumPy broadcasting + (a-b)^2=a^2-2ab+b^2
    """
    dm = np.sqrt((s1 * s1).sum(1)[:, None] - 2.0 * np.dot(s1, s2.T) + (s2 * s2).sum(1))
    return 0.5 * (dm.min(0).mean() + dm.min(1).mean())


def avg_mam_distance_numpy_faster(s1, s2):
    """NumPy broadcasting + less sqrt() evaluations
    """
    dm = (s1 * s1).sum(1)[:, None] - 2.0 * np.dot(s1, s2.T) + (s2 * s2).sum(1)
    return 0.5 * (np.sqrt(dm.min(0)).mean() + np.sqrt(dm.min(1)).mean())


def avg_mam_distance_numpy_faster2(s1, s2):
    """Scipy outer + less sqrt() evaluations
    """
    dmx = np.subtract.outer(s1[:,0], s2[:,0])
    dmy = np.subtract.outer(s1[:,1], s2[:,1])
    dmz = np.subtract.outer(s1[:,2], s2[:,2])
    dm = dmx * dmx + dmy * dmy + dmz * dmz
    return 0.5 * (np.sqrt(dm.min(0)).mean() + np.sqrt(dm.min(1)).mean())


def avg_mam_distance_numpy_faster3(s1, s2):
    """sklearn pairwise_distances() + less sqrt evaluations
    """
    dm = pairwise_distances(s1, s2, metric='sqeuclidean')
    return 0.5 * (np.sqrt(dm.min(0)).mean() + np.sqrt(dm.min(1)).mean())


def avg_mam_distance_numpy_faster4(s1, s2):
    """Just NumPy broadcasting over each dimension
    """
    dmx = s1[:,0][:, None] - s2[:,0]
    dmy = s1[:,1][:, None] - s2[:,1]
    dmz = s1[:,2][:, None] - s2[:,2]
    dm = dmx * dmx + dmy * dmy + dmz * dmz
    return 0.5 * (np.sqrt(dm.min(0)).mean() + np.sqrt(dm.min(1)).mean())


def avg_mam_distance_numpy_faster5(s1, s2):
    """KDTree construction + query
    """
    kdt1 = KDTree(s1)
    kdt2 = KDTree(s2)
    return 0.5 * (kdt2.query(s1, k=1)[0].mean() + kdt1.query(s2, k=1)[0].mean())


def avg_mam_distance_numpy_faster6(s1, s2, kdt1, kdt2):
    """Pre-built KDTrees then query
    """
    return 0.5 * (kdt2.query(s1, k=1)[0].mean() + kdt1.query(s2, k=1)[0].mean())


def avg_mam_distance_numpy_faster7(s1, s2):
    """Just NumPy with very compact broadcasting
    """
    dm = s1[:, None, :] - s2[None, :, :]
    dm = (dm * dm).sum(2)
    return 0.5 * (np.sqrt(dm.min(0)).mean() + np.sqrt(dm.min(1)).mean())


if __name__ == '__main__':

    np.random.seed(0)

    s1 = np.random.random((100,3))
    s2 = np.random.random((200,3))

    distances= [avg_mam_distance_dipy,
                avg_mam_distance_scipy,
                avg_mam_distance_numpy,
                avg_mam_distance_numpy_faster,
                avg_mam_distance_numpy_faster2,
                avg_mam_distance_numpy_faster3,
                avg_mam_distance_numpy_faster4,
                avg_mam_distance_numpy_faster5,
                avg_mam_distance_numpy_faster7,
                ]

    for i, distance in enumerate(distances):
        print i, ')', distance.__doc__.strip() + '(s1, s2) =', distance(s1, s2)

    kdt1 = KDTree(s1)
    kdt2 = KDTree(s2)
    print i+1, ") avg_mam_distance_numpy_faster6(s1, s2) =", avg_mam_distance_numpy_faster6(s1, s2, kdt1, kdt2)

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

    dms = []
    for k, distance in enumerate(distances):
        print k, ')', distance.__doc__.strip(), ':',
        t0 = time()
        dm = np.zeros((len(B1), len(B2)))
        for i in range(len(B1)):
            for j in range(len(B2)):
                dm[i, j] = distance(B1[i], B2[j])
                
        print time() - t0 , 'sec.'
        dms.append(dm)
        if len(dms) > 1:
            np.testing.assert_almost_equal(dms[-1], dms[-2], decimal=5)
                         

    print k+1, ') ' + avg_mam_distance_numpy_faster6.__doc__.strip() + ':', 
    t0 = time()
    mam_dm_python_numpy_faster6 = np.zeros((len(B1), len(B2)))
    kdtB2 = [KDTree(s2) for s2 in B2]
    for i in range(len(B1)):
        kdt1 = KDTree(B1[i])
        for j in range(len(B2)):
            mam_dm_python_numpy_faster6[i, j] = avg_mam_distance_numpy_faster6(B1[i], B2[j], kdt1, kdtB2[j])

    print time() - t0 , 'sec.'
    np.testing.assert_almost_equal(dms[-1], mam_dm_python_numpy_faster6, decimal=5)

    

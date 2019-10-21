import numpy as np

data_sift = np.load('voc_sift_kmeans_20.npy')
data_surf = np.load('voc_surf_kmeans_20.npy')
data_orb = np.load('voc_orb_kmeans_20.npy')
print(data_sift)
print(data_surf)
print(data_orb)
hac_sift = np.load('voc_sift_hierarchical_20.npy')
hac_surf = np.load('voc_surf_hierarchical_20.npy')
hac_orb = np.load('voc_orb_hierarchical_20.npy')
print(hac_sift)
print(hac_surf)
print(hac_orb)


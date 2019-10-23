import numpy as np

for i in ['sift','surf','orb']:
	for j in ['kmeans', 'hierarchical']:
		for k in ['20','50']:
			data = np.load('voc_' + i + '_' + j + '_' + k + '.npy')
			print(i+'_'+j + '_'+k)
			print(data)
			print()

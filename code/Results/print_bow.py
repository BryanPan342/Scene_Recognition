import numpy as np
for i in range(12):
	for j in ['train', 'test']:
		bow_test = np.load('bow_' + j + '_' + str(i) + '.npy')
		print()
		print('bow_' + j + '_' + str(i))
		print(bow_test)

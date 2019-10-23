import numpy as np
for i in ['knn','lin','rbf']:
	print()
	print(i)
	data = np.load(i+'_accuracies.npy')
	print(data)


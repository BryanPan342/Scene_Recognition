import os
import cv2
import numpy as np
import timeit, time
from sklearn import neighbors, svm, cluster, preprocessing
from random import sample


def load_data():
    test_path = '../data/test/'
    train_path = '../data/train/'
    
    train_classes = sorted([dirname for dirname in os.listdir(train_path)], key=lambda s: s.upper())
    test_classes = sorted([dirname for dirname in os.listdir(test_path)], key=lambda s: s.upper())
    train_labels = []
    test_labels = []
    train_images = []
    test_images = []
    for i, label in enumerate(train_classes):
        for filename in os.listdir(train_path + label + '/'):
            image = cv2.imread(train_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            train_images.append(image)
            train_labels.append(i)
    for i, label in enumerate(test_classes):
        for filename in os.listdir(test_path + label + '/'):
            image = cv2.imread(test_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            test_images.append(image)
            test_labels.append(i)
            
    return train_images, test_images, train_labels, test_labels


def KNN_classifier(train_features, train_labels, test_features, num_neighbors):
    # outputs labels for all testing images

    # train_features is an N x d matrix, where d is the dimensionality of the
    # feature representation and N is the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer
    # indicating the ground truth category for each training image.
    # test_features is an M x d array, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # num_neighbors is the number of neighbors for the KNN classifier

    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test image.

    knn = neighbors.KNeighborsClassifier(n_neighbors = num_neighbors, algorithm='kd_tree',metric='euclidean')
    knn.fit(train_features, train_labels)
    predicted_categories = knn.predict(test_features)
    return predicted_categories


def SVM_classifier(train_features, train_labels, test_features, is_linear, svm_lambda):
    # this function will train a linear svm for every category (i.e. one vs all)
    # and then use the learned linear classifiers to predict the category of
    # every test image. every test feature will be evaluated with all 15 svms
    # and the most confident svm will "win". confidence, or distance from the
    # margin, is w*x + b where '*' is the inner product or dot product and w and
    # b are the learned hyperplane parameters.

    # train_features is an N x d matrix, where d is the dimensionality of
    # the feature representation and N the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer 
    # indicating the ground truth category for each training image.
    # test_features is an M x d matrix, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # is_linear is a boolean. If true, you will train linear SVMs. Otherwise, you 
    # will use SVMs with a Radial Basis Function (RBF) Kernel.
    # svm_lambda is a scalar, the value of the regularizer for the SVMs

    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test feature.

    predicted_categories = []
    classifier_list = []
    for i in range(15):
        temp=np.copy(train_labels)
        for j in range(len(train_labels)):
            if train_labels[j] == i:
                temp[j] = i
            else:
                temp[j] = 15
        if is_linear:
            classifier = svm.LinearSVC(C=svm_lambda).fit(train_features, temp)
        else:
            classifier = svm.SVC(kernel='rbf', C=svm_lambda, gamma='scale').fit(train_features, temp)
        classifier_list.append(classifier)
    for i in test_features:
        predicted_categories.append(np.asarray([c.decision_function([i])[0] for c in classifier_list]).argmin())
    # classifier = svm.LinearSVC()
    # predicted_categories = classifier.fit(train_features, train_labels).predict(test_features)
    # if is_linear:
    #     classifier = svm.LinearSVC()
    # else:
    #     classifier = svm.SVC(kernel='rbf', gamma=svm_lambda)
    # fit = classifier.fit(train_features, train_labels)
    # predicted_categories = fit.predict(test_features)

    return predicted_categories


def imresize(input_image, target_size):
    # resizes the input image, represented as a 2D array, to a new image of size [target_size, target_size]. 
    # Normalizes the output image to be zero-mean, and in the [-1, 1] range.
    resized_image = cv2.resize(input_image, (target_size ,target_size))
    output_image = cv2.normalize(resized_image, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F, alpha=-1, beta=1)
    return np.reshape(output_image, (-1))

def reportAccuracy(true_labels, predicted_labels):
    # generates and returns the accuracy of a model

    # true_labels is a N x 1 list, where each entry is an integer
    # and N is the size of the testing set.
    # predicted_labels is a N x 1 list, where each entry is an 
    # integer, and N is the size of the testing set. These labels 
    # were produced by your system.

    # accuracy is a scalar, defined in the spec (in %)
    accuracy = 0
    for i in range(len(true_labels)):
        if true_labels[i] == predicted_labels[i]:
            accuracy+=1
    return accuracy * 100. / len(true_labels)


def buildDict(train_images, dict_size, feature_type, clustering_type):
    # this function will sample descriptors from the training images,
    # cluster them, and then return the cluster centers.

    # train_images is a list of N images, represented as 2D arrays
    # dict_size is the size of the vocabulary,
    # feature_type is a string specifying the type of feature that we are interested in.
    # Valid values are "sift", "surf" and "orb"
    # clustering_type is one of "kmeans" or "hierarchical"

    # the output 'vocabulary' should be a list of length dict_size, with elements of size d, where d is the 
    # dimention of the feature. each row is a cluster centroid / visual word.

    # NOTE: Should you run out of memory or have performance issues, feel free to limit the 
    # number of descriptors you store per image.
    features = []
    vocabulary = [None for i in range(dict_size)]
    count = [0 for i in range(dict_size)]
    if feature_type == "sift":
        transformation = cv2.xfeatures2d.SIFT_create(nfeatures=28)
    elif feature_type == "surf":
        transformation = cv2.xfeatures2d.SURF_create(extended=False)
    elif feature_type == "orb":
        transformation = cv2.ORB_create(nfeatures=28)
    for i in range(len(train_images)):
        kp, descriptors = transformation.detectAndCompute(train_images[i], None)
        if feature_type=='surf' and len(descriptors) > 28:            #surf version of nfeatures=25 to reduce number of features per image
            descriptors = descriptors[np.random.randint(descriptors.shape[0], size=28), :]
        for des in descriptors:
            features.append(des)
    features = np.asarray(features)
    if clustering_type == "kmeans":
        model = cluster.KMeans(n_clusters=dict_size, n_jobs = -1)
        vocabulary = model.fit(features).cluster_centers_
    elif clustering_type == "hierarchical":
        model = cluster.AgglomerativeClustering(n_clusters=dict_size)
        labels = model.fit(features).labels_   #fit provides a label for each element in features to which centroid it belongs to
        for i in range(len(labels)):           #create a vocab for every centroid by adding up all the descriptors per image
            if count[labels[i]]!=0:
                vocabulary[labels[i]] = np.add(vocabulary[labels[i]], features[i])
            else:
                vocabulary[labels[i]] = features[i].astype(float)
            count[labels[i]] += 1
        vocabulary = np.asarray(vocabulary)
        for i in range(len(vocabulary)):    #average all 
            vocabulary[i] = np.divide(vocabulary[i], count[i])
    return vocabulary

def computeBow(image, vocabulary, feature_type):
    # extracts features from the image, and returns a BOW representation using a vocabulary

    # image is 2D array
    # vocabulary is an array of size dict_size x d
    # feature type is a string (from "sift", "surf", "orb") specifying the feature
    # used to create the vocabulary
    if feature_type == "sift":
        transformation = cv2.xfeatures2d.SIFT_create()
    elif feature_type == "surf":
        transformation = cv2.xfeatures2d.SURF_create()
    elif feature_type == "orb":
        transformation = cv2.ORB_create()
    kp, descriptors = transformation.detectAndCompute(image, None)
    Bow=[0 for i in vocabulary]
    for i in descriptors:
        Bow[np.array(np.linalg.norm(i-vocabulary, axis=1)).argmin()]+=1

    # BOW is the new image representation, a normalized histogram
    return np.asarray(Bow) / float(len(descriptors))


def tinyImages(train_features, test_features, train_labels, test_labels):
    # Resizes training images and flattens them to train a KNN classifier using the training labels
    # Classifies the resized and flattened testing images using the trained classifier
    # Returns the accuracy of the system, and the overall runtime (including resizing and classification)
    # Does so for 8x8, 16x16, and 32x32 images, with 1, 3 and 6 neighbors

    # train_features is a list of N images, represented as 2D arrays
    # test_features is a list of M images, represented as 2D arrays
    # train_labels is a list of N integers, containing the label values for the train set
    # test_labels is a list of M integers, containing the label values for the test set

    # classResult is a 18x1 array, containing accuracies and runtimes, in the following order:
    # accuracies and runtimes for 8x8 scales, 16x16 scales, 32x32 scales
    # [8x8 scale 1 neighbor accuracy, 8x8 scale 1 neighbor runtime, 8x8 scale 3 neighbor accuracy, 
    # 8x8 scale 3 neighbor runtime, ...]
    # Accuracies are a percentage, runtimes are in seconds
    
    arr_pix = np.array([8,16,32])
    arr_n = np.array([1,3,6])
    
    classResult = []
    for i in range(len(arr_pix)):
        resized_test = []
        resized_train = []
        t1 = time.time()
        for j in range(len(train_features)):
            resized_train.append(imresize(train_features[j], arr_pix[i]))
        for l in range(len(test_features)):
            resized_test.append(imresize(test_features[l], arr_pix[i]))
        # print(np.ndarray(train_features).tolist())
        for k in range(len(arr_n)):
            predicted_labels = KNN_classifier(resized_train, train_labels, resized_test, arr_n[k])
            classResult.append(reportAccuracy(test_labels, predicted_labels))
            classResult.append(time.time()-t1) 
    return classResult
    
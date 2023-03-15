import cv2 as cv
import numpy as np
import os

best_acc_for_vocab = []
words = [10, 25, 50, 75, 100, 250, 500, 1000, 2500, 5000]
for w in range(len(words)):
    print("Number of words used for the vocabulary:", words[w])
    folder = 'vocabulary_files'
    folder_in_folder = 'vocabulary_with_' + str(words[w]) + '_words'
    path1 = os.path.join(folder, folder_in_folder)

    file = 'index.npy'
    path = os.path.join(path1, file)
    bow_descs = np.load(path).astype(np.float32)

    file = 'paths.npy'
    path = os.path.join(path1, file)
    img_paths = np.load(path)

    file = 'vocabulary.npy'
    path = os.path.join(path1, file)
    vocabulary = np.load(path)

    file = 'index_test.npy'
    path = os.path.join(path1, file)
    test_descs = np.load(path).astype(np.float32)

    file = 'paths_test.npy'
    path = os.path.join(path1, file)
    img_paths_test = np.load(path)

    # Labels
    labels = []
    for p in img_paths:
        if '145.motorbikes-101' in p:
            labels.append(0)
        elif '178.school-bus' in p:
            labels.append(1)
        elif '224.touring-bike' in p:
            labels.append(2)
        elif '251.airplanes-101' in p:
            labels.append(3)
        elif '252.car-side-101' in p:
            labels.append(4)
    labels = np.array(labels, np.int32)

    n = 5
    num_cl_train = [0, 0, 0, 0, 0]

    for i in range(n):
        for j in range(len(labels)):
            if labels[j] == i:
                num_cl_train[i] += 1

    # labels for one versus all
    labels_train_svm = []
    for c in range(n):
        l = []
        for el in labels:
            if el == c:
                l.append(1)
            else:
                l.append(0)
        labels_train_svm.append(l)

    labels_train_svm = np.array(labels_train_svm, np.int32)

    kernel_types = {"RBF":cv.ml.SVM_RBF, "LINEAR":cv.ml.SVM_LINEAR, "SIGMOID":cv.ml.SVM_SIGMOID,
                    "CHI2":cv.ml.SVM_CHI2, "INTER":cv.ml.SVM_INTER}

    folder = 'svm_files'
    folder_in_folder = 'svm_with_' + str(words[w]) + '_words'
    path1 = os.path.join(folder, folder_in_folder)
    os.mkdir(path1)

    for key, value in kernel_types.items():
        print("\nKernel:", key)
        # Train SVM one versus all approach
        for i in range(n):
            print('Training SVM' + str(i))
            svm = cv.ml.SVM_create()
            svm.setType(cv.ml.SVM_C_SVC)
            svm.setKernel(value)
            svm.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))

            svm.trainAuto(bow_descs, cv.ml.ROW_SAMPLE, labels_train_svm[i])

            file = key + '_svm' + str(i)
            path_svm = os.path.join(path1, file)
            svm.save(path_svm)
    print("------------------------------------------------------------------------------------------------------------------------------")
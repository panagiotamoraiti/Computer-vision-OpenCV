import cv2 as cv
import numpy as np
import os

# For one image
def svm_func(bow_desc, labels, bow_descs, svm_load):
    response = []
    for i in range(len(svm_load)):
        r = svm_load[i].predict(bow_desc, flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
        response.append(r[1])

    result = np.argmin(response)

    return result


# For many images
def svm_test(test_descs, labels, bow_descs, svm_load):
    results = []
    for i in range(len(test_descs)):
        test = np.resize(test_descs[i], (1, test_descs[i].shape[0]))
        result = svm_func(test, labels, bow_descs, svm_load)
        results.append(result)
    results = np.array(results, np.int32)

    return results


def show_img_and_prediction(results, test_descs, img_paths_test):
    classes = ["motorbike", "school-bus", "touring-bike", "airplane", "car-side"]
    num_cl = [0, 0, 0, 0, 0]
    n = 5
    for i in range(n):
        for j in range(len(labels_test)):
            if labels_test[j] == i:
                num_cl[i] += 1

    for c in range(len(num_cl)):
        name = "\nClass" + str(c)
        print(name)

        cv.namedWindow(name, cv.WINDOW_NORMAL)
        for i in range(len(labels_test)):
            if labels_test[i] == c:
                prediction = classes[results[i]]
                print("Test image " + str(i+1) + ": " + "It is a " + prediction)
                test_img = cv.imread(img_paths_test[i])
                cv.imshow(name, test_img)
                cv.waitKey(0)


def accuracy(labels_test, predictions, show_all = True):
    # Total
    matches = np.count_nonzero(labels_test == predictions)
    total_acc = matches / len(labels_test) * 100
    total_acc = round(total_acc, 2)
    if (show_all):
        print("Correctly predicted: " + str(matches) + " images out of " + str(len(labels_test)) + " images")
        print("Total accuracy: " + str(total_acc) + "%\n")

    # For every class
    num_cl = [0, 0, 0, 0, 0]
    n = 5
    matches = [0, 0, 0, 0, 0]
    for i in range(n):
        for j in range(len(labels_test)):
            if labels_test[j] == i:
                if labels_test[j] == predictions[j]:
                    matches[i] += 1
                num_cl[i] += 1

    acc = []
    for i in range(n):
        acc.append(round(matches[i] / num_cl[i] * 100, 2))

    if (show_all):
        for i in range(n):
            print("Correctly predicted images for class" + str(i) + ": " + str(matches[i])+ " out of " + str(num_cl[i]) + " images")
            print("Accuracy for class"+ str(i) + " is: " + str(acc[i]) + "%\n")

    return total_acc


best_acc_for_vocab = []
kernel_type = []
words = [10, 25, 50, 75, 100, 250, 500, 1000, 2500, 5000]
for w in range(len(words)):
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

    # Labels test
    labels_test = []
    for p in img_paths_test:
        if '145.motorbikes-101' in p:
            labels_test.append(0)
        elif '178.school-bus' in p:
            labels_test.append(1)
        elif '224.touring-bike' in p:
            labels_test.append(2)
        elif '251.airplanes-101' in p:
            labels_test.append(3)
        elif '252.car-side-101' in p:
            labels_test.append(4)
    labels_test = np.array(labels_test, np.int32)


    print("Number of words used for the vocabulary: " + str(words[w]) + '\n')

    # Load SVM
    kernel_types = {"RBF": cv.ml.SVM_RBF, "LINEAR": cv.ml.SVM_LINEAR, "SIGMOID": cv.ml.SVM_SIGMOID,
                    "CHI2": cv.ml.SVM_CHI2, "INTER": cv.ml.SVM_INTER}
    n = 5

    folder = 'svm_files'
    folder_in_folder = 'svm_with_' + str(words[w]) + '_words'
    path1 = os.path.join(folder, folder_in_folder)

    acc_kernel = {}

    for key, value in kernel_types.items():
        print("Kernel:", key)
        svm_load = []

        for i in range(n):
            path = os.path.join(path1, key + '_svm' + str(i))
            svm_obj = cv.ml.SVM_create()
            svm_obj = svm_obj.load(path)
            svm_load.append(svm_obj)

        # SVM for all test images
        results = svm_test(test_descs, labels_test, bow_descs, svm_load)

        # Print total accuracy and accuracy for every class
        acc = accuracy(labels_test, results, show_all=False)
        acc_kernel[key] = acc
        print("Accuracy: " + str(acc) + "%\n")

    print('' + str(acc_kernel) + '\n')
    best_kernel = max(acc_kernel, key=acc_kernel.get)
    best_acc_for_vocab.append(max(acc_kernel.values()))
    kernel_type.append(best_kernel)

    print("Max accuracy: " + str(max(acc_kernel.values())) + "%")
    print("Kernel:", best_kernel)
    print("------------------------------------------------------------------------------------------------------------------------------")

print(best_acc_for_vocab)
print(kernel_type)
print("------------------------------------------------------------------------------------------------------------------------------")

words_best = best_acc_for_vocab.index(max(best_acc_for_vocab))
best_kernel = best_acc_for_vocab.index(max(best_acc_for_vocab))

print("Max accuracy: " + str(max(best_acc_for_vocab)) + "%" + " for kernel " + str(kernel_type[best_kernel]) +
      " and " +  str(words[words_best]) + " words.")

# Best BOW
folder = "vocabulary_files"
folder_inside = 'vocabulary_with_' + str(words[words_best]) + '_words'
path1 = os.path.join(folder, folder_inside)

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

folder = 'svm_files'
folder_in_folder = 'svm_with_' + str(words[words_best]) + '_words'
path1 = os.path.join(folder, folder_in_folder)

# Best accuracy for kernel
best_kernel =  kernel_type[best_kernel]

print("\nBest accuracy")
svm_load = []
folder = 'svm_files'
folder_in_folder = 'svm_with_' + str(words[words_best]) + '_words'
path1 = os.path.join(folder, folder_in_folder)

for i in range(n):
    file = best_kernel + '_svm' + str(i)
    path = os.path.join(path1, file)

    svm_obj = cv.ml.SVM_create()
    svm_obj = svm_obj.load(path)
    svm_load.append(svm_obj)

results = svm_test(test_descs, labels_test, bow_descs, svm_load)
acc = accuracy(labels_test, results)

# Show prediction and image for every test image
show_img_and_prediction(results, test_descs, img_paths_test)

# Write best parameters to a txt file
# folder = "best_parameters"
# file = "SVM_best_parameters.txt"
# path = os.path.join(folder, file)
# lines = [str(words[words_best]), str(best_kernel)]
# with open(path, 'w') as f:
#     for line in lines:
#         f.write(line)
#         f.write('\n')
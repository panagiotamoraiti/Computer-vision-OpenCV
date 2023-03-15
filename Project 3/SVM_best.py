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

# Best BOW
folder = "best_parameters"
file = "SVM_best_parameters.txt"
path = os.path.join(folder, file)

with open(path) as f:
    lines = f.read().splitlines()

words = int(lines[0])
best_kernel = lines[1]

folder = "vocabulary_files"
folder_inside = 'vocabulary_with_' + str(words) + '_words'
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
folder_in_folder = 'svm_with_' + str(words) + '_words'
path1 = os.path.join(folder, folder_in_folder)

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

# Best accuracy for kernel
print("Best accuracy for " + best_kernel +
      " kernel and " +  str(words) + " words\n")

svm_load = []
folder = 'svm_files'
folder_in_folder = 'svm_with_' + str(words) + '_words'
path1 = os.path.join(folder, folder_in_folder)

n = 5

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
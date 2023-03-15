import cv2 as cv
import numpy as np
import os

# For one image
def knn(bow_desc, labels, bow_descs, neigh_num):
    distances = np.sum((bow_desc - bow_descs) ** 2, axis=1)
    # distances = np.sqrt(np.sum((bow_desc - bow_descs) ** 2, axis=1))  # euclidian distance
    # distances = np.sum(np.abs(bow_desc - bow_descs), axis=1) # manhattan distance
    retrieved_ids = np.argsort(distances)
    ids = retrieved_ids.tolist()

    categories = [0, 0, 0, 0, 0]
    for i in range(neigh_num):
        categories[labels[ids[i]]] += 1
    result = categories.index(max(categories))

    return result


# For many images
def knn_test(test_descs, labels, bow_descs, neigh_num):
    results = []
    for i in range(len(test_descs)):
        result = knn(test_descs[i], labels, bow_descs, neigh_num)
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
file = "KNN_best_parameters.txt"
path = os.path.join(folder, file)

with open(path) as f:
    lines = f.read().splitlines()

words = int(lines[0])
n = int(lines[1])

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


# Best accuracy for n neighbours
print("Best accuracy for " + str(n) +
      " neighbours and " +  str(words) + " words\n")

results = knn_test(test_descs, labels, bow_descs, n)
acc = accuracy(labels_test, results)

# Show prediction and image for every test image
show_img_and_prediction(results, test_descs, img_paths_test)
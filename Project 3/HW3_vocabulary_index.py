import os
import cv2 as cv
import numpy as np

train_folders = ['imagedb/145.motorbikes-101', 'imagedb/178.school-bus', 'imagedb/224.touring-bike',
                 'imagedb/251.airplanes-101', 'imagedb/252.car-side-101']

sift = cv.xfeatures2d_SIFT.create()


def extract_local_features(path):
    img = cv.imread(path)
    kp = sift.detect(img)
    desc = sift.compute(img, kp)
    desc = desc[1]
    return desc


# Extract Database
print('Extracting features...')
train_descs = np.zeros((0, 128))
for folder in train_folders:
    files = os.listdir(folder)
    for file in files:
        path = os.path.join(folder, file)
        desc = extract_local_features(path)
        if desc is None:
            continue
        train_descs = np.concatenate((train_descs, desc), axis=0)

# Create vocabulary
words = [10, 25, 50, 75, 100, 250, 500, 1000, 2500, 5000]
for w in range(len(words)):
    print('Creating vocabulary...' + str(w+1))
    term_crit = (cv.TERM_CRITERIA_EPS, 30, 0.1)
    trainer = cv.BOWKMeansTrainer(words[w], term_crit, 1, cv.KMEANS_PP_CENTERS)
    vocabulary = trainer.cluster(train_descs.astype(np.float32))

    # Make a new folder for every vocabulary saved
    folder = 'vocabulary_files'
    folder_in_folder = 'vocabulary_with_' + str(words[w]) + '_words'
    path1 = os.path.join(folder, folder_in_folder)
    os.mkdir(path1)

    file = 'vocabulary.npy'
    path1 = os.path.join(path1, file)
    np.save(path1, vocabulary)

    print('Creating index...')
    # Classification
    descriptor_extractor = cv.BOWImgDescriptorExtractor(sift, cv.BFMatcher(cv.NORM_L2SQR))
    descriptor_extractor.setVocabulary(vocabulary)

    img_paths = []
    # train_descs = np.zeros((0, 128))
    bow_descs = np.zeros((0, vocabulary.shape[0]))
    for folder in train_folders:
        files = os.listdir(folder)
        for file in files:
            path = os.path.join(folder, file)

            img = cv.imread(path)
            kp = sift.detect(img)
            bow_desc = descriptor_extractor.compute(img, kp)

            img_paths.append(path)
            bow_descs = np.concatenate((bow_descs, bow_desc), axis=0)

    path1 = folder = 'vocabulary_files'
    folder_in_folder = 'vocabulary_with_' + str(words[w]) + '_words'
    path1 = os.path.join(folder, folder_in_folder)

    file = 'index.npy'
    path1 = os.path.join(path1, file)
    np.save(path1, bow_descs)

    path1 = folder = 'vocabulary_files'
    folder_in_folder = 'vocabulary_with_' + str(words[w]) + '_words'
    path1 = os.path.join(folder, folder_in_folder)

    file = 'paths'
    path1 = os.path.join(path1, file)
    np.save(path1, img_paths)

    test_folders = ['imagedb_test/145.motorbikes-101', 'imagedb_test/178.school-bus', 'imagedb_test/224.touring-bike',
                     'imagedb_test/251.airplanes-101', 'imagedb_test/252.car-side-101']

    img_paths_test = []
    test_descs = np.zeros((0, vocabulary.shape[0]))
    for folder in test_folders:
        files = os.listdir(folder)
        for file in files:
            path = os.path.join(folder, file)
            img_paths_test.append(path)
            img = cv.imread(path)
            kp = sift.detect(img)
            test_desc = descriptor_extractor.compute(img, kp)
            test_descs = np.concatenate((test_descs, test_desc), axis=0)

    path1 = folder = 'vocabulary_files'
    folder_in_folder = 'vocabulary_with_' + str(words[w]) + '_words'
    path1 = os.path.join(folder, folder_in_folder)

    file = 'index_test.npy'
    path1 = os.path.join(path1, file)
    np.save(path1, test_descs)

    path1 = folder = 'vocabulary_files'
    folder_in_folder = 'vocabulary_with_' + str(words[w]) + '_words'
    path1 = os.path.join(folder, folder_in_folder)

    file = 'paths_test'
    path1 = os.path.join(path1, file)
    np.save(path1, img_paths_test)
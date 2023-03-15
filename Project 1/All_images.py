import cv2
import numpy as np
import os.path

def median_fiter(image, kernel_size):
    image_median = image.copy()
    k = kernel_size // 2
    image_median_boarder = cv2.copyMakeBorder(image_median, k, k, k, k, cv2.BORDER_REFLECT)
    for i in range(k, image_median_boarder.shape[0]-k):
        for j in range(k, image_median_boarder.shape[1]-k):
            kernel = image_median_boarder[i-k:i+kernel_size-k, j-k:j+kernel_size-k]
            median_value = np.median(kernel)
            image_median[i-k, j-k] = median_value
    return image_median

def my_integral(image):
    integral_image = image.copy()
    integral_image = np.concatenate((np.zeros((1, image.shape[1])), integral_image), axis=0)
    integral_image = np.concatenate((np.zeros((image.shape[0]+1, 1)), integral_image), axis=1)
    integral_image = integral_image.astype(int)
    integral_image = np.cumsum(integral_image, axis = 1)
    integral_image = np.cumsum(integral_image, axis = 0)
    return integral_image

for i in range(10):
    filename = 'noise/' + str(i) + '.png'
    noise_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # cv2.namedWindow('noise')
    # cv2.imshow('noise', noise_img)
    # cv2.waitKey(0)

    # Filter salt and pepper noise using median filter
    filename = 'All_filtered_and_final_images/filtered_img' + str(i) + '.png'
    if not os.path.exists(filename):
        filtered_img = median_fiter(noise_img, 3)
        cv2.imwrite(filename, filtered_img)
    filtered_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # cv2.namedWindow('filtered', cv2.WINDOW_NORMAL)
    # cv2.imshow('filtered', filtered_img)
    # cv2.waitKey(0)

    # Convert image from grayscale to binary
    thr, binary_img = cv2.threshold(filtered_img, 6, 255, cv2.THRESH_BINARY)
    # cv2.namedWindow('binary1')
    # cv2.imshow('binary1', binary_img)
    # cv2.waitKey(0)

    # Closing
    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, strel)
    # cv2.namedWindow('binary closing')
    # cv2.imshow('binary closing', binary_img)
    # cv2.waitKey(0)

    # Opening
    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13,13))
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, strel)
    # cv2.namedWindow('binary opening')
    # cv2.imshow('binary opening', binary_img)
    # cv2.waitKey(0)

    # Red color in BGR
    color = (0, 0, 255)
    # Line thickness of 2 pixels
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5

    # Integral
    int_img = my_integral(filtered_img)

    # Find connected components and statistics
    n_labels, labels = cv2.connectedComponents(binary_img)
    filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)

    for label in range(1, n_labels):
        mask = np.array(labels, dtype=np.uint8)
        mask[labels == label] = 255
        mask[labels != label] = 0
        x, y, w, h = cv2.boundingRect(mask)
        filtered_img = cv2.rectangle(filtered_img, (x, y),(x+w, y+h), color, thickness)
        org = (int(x+0.1*w), int(y+h*0.9))
        filtered_img = cv2.putText(filtered_img, str(label), org, font, fontScale, color, thickness, cv2.LINE_AA)

        area = np.count_nonzero(mask == 255)
        bounding_box_area = w*h
        mean_graylevel_value = (int_img[y, x] - int_img[h+y, x] - int_img[y, w+x] + int_img[h+y, w+x])/bounding_box_area

        print("---- Region " + str(label) + ": ----")
        print("Area (px):", area)
        print("Bounding Box Area (px):", bounding_box_area)
        print("Mean graylevel value in bounding box:", mean_graylevel_value)

    # Save the final image
    filename = 'All_filtered_and_final_images/final_img_filtered' + str(i) + '.png'
    if not os.path.exists(filename):
        cv2.imwrite(filename, filtered_img)

    final_img = cv2.imread(filename)
    cv2.namedWindow('final', cv2.WINDOW_NORMAL)
    cv2.imshow('final', final_img)
    cv2.waitKey(0)

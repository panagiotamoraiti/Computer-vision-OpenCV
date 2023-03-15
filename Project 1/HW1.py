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

filename = 'original/4.png'
original_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
# cv2.namedWindow('original')
# cv2.imshow('original', original_img)
# cv2.waitKey(0)

filename = 'noise/4.png'
noise_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
# cv2.namedWindow('noise')
# cv2.imshow('noise', noise_img)
# cv2.waitKey(0)

# Filter salt and pepper noise using median filter
filename = 'filtered/filtered_img.png'
if not os.path.exists(filename):
    filtered_img = median_fiter(noise_img, 3)
    cv2.imwrite(filename, filtered_img)
filtered_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
cv2.namedWindow('filtered', cv2.WINDOW_NORMAL)
cv2.imshow('filtered', filtered_img)
cv2.waitKey(0)

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

# cv2.namedWindow('binary preprocessed', cv2.WINDOW_NORMAL)
# cv2.imshow('binary preprocessed', binary_img)
# cv2.waitKey(0)

# Closing
strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, strel, iterations=2)
# cv2.namedWindow('binary closing2')
# cv2.imshow('binary closing2', binary_img)
# cv2.waitKey(0)

# Opening
strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13 ,13))
binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, strel)
# cv2.namedWindow('binary opening2')
# cv2.imshow('binary opening2', binary_img)
# cv2.waitKey(0)

cv2.namedWindow('binary preprocessed2', cv2.WINDOW_NORMAL)
cv2.imshow('binary preprocessed2', binary_img)
cv2.waitKey(0)

# Red color in BGR
color = (0, 0, 255)
# Line thickness of 2 pixels
thickness = 1
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5

# Integral
filename = 'summed_area_table.npy'
if not os.path.exists(filename):
    int_img = my_integral(filtered_img)
    np.save('summed_area_table.npy', int_img)
int_img = np.load(filename)

# Check if summed_area_table and cv2.integral are the same
# opencv_int_img = cv2.integral(filtered_img)
# print((int_img == opencv_int_img).all())

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

# cv2.namedWindow('connected components', cv2.WINDOW_NORMAL)
# cv2.imshow('connected components', filtered_img)
# cv2.waitKey(0)

# Save the final image
filename = 'final/final_img_filtered.png'
if not os.path.exists(filename):
    cv2.imwrite(filename, filtered_img)

final_img = cv2.imread(filename)
cv2.namedWindow('final', cv2.WINDOW_NORMAL)
cv2.imshow('final', final_img)
cv2.waitKey(0)

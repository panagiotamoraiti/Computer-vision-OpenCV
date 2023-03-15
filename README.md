# Computer-vision-OpenCV

Project 1
Implementation of the median filter, in order to remove salt and pepper noise from the images.

Detect all cells in microscope images. The program displays and saves the image, in which the different cells are depicted. 
Specifically, for each cell there is a surrounding box (bounding box) and a unique serial number.

For each cell we measure:
1. The area of the cell, defined as the number of pixels that belongs to the cell.
2. The area of the cell's bounding box, as the number of pixels that belongs to the bounding box.
3. The average grayscale value of the pixels contained in the bounding box, in such a way that the execution speed of the calculation 
is independent of the size of the sub-region. To resolve this question, we do not use the 'cv2.integral' function.


Project 2
Implementation of an algorithm that produces panoramas from multiple subjects pictures (we use 4 images). 
We use SIFT, SURF and Image Composite Editor, in order to compare the results.


Project 3
We use OpenCV library, in order to solve a multi-class classification problem. 

The program is implemented by performing the following steps:
1. Visual vocabulary production based on the Bag of Visual Words model (BOVW). 
The creation of the dictionary should be done using the K-Means algorithm using all images in the training set (imagedb_train).
2. Extract a descriptor on each training image (imagedb_train) based on BOVW model using the dictionary generated in step 1.
3. Based on the results of step 2, implement the classification function of an image using the following two classifiers:
  a. k-NN algorithm without using the associated OpenCV function (cv.ml.KNearest_create()).
  b. one-versus-all method where for each class an SVM classifier is trained.
4. System evaluation: Using the test set (imagedb_test), measure the accuracy of the system (in both cases of classifiers) expressed 
as the percentage of successful classifications, overall and by class.

Project 4
We implement a convolutional network architecture in Python using the Keras Tensorflow library, which will address the problem 
of multi-class classification.
As part of the work, two architectures are implemented:
  1. A NON-pre-trained network which is created exclusively for the current one classification problem.
  3. A pre-trained network. ResNet 50 is used.
  
 Best models and datasets can be found following the link: 
 https://drive.google.com/drive/folders/1nCnI0R7ONX1QePdz0qL76360luyyLpKu?usp=sharing 

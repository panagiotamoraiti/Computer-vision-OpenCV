import numpy as np
import cv2 as cv
import os.path

sift = cv.xfeatures2d_SIFT.create()
surf = cv.xfeatures2d_SURF.create()

# Read images
img1 = cv.imread('rio/rio-01.png', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('rio/rio-02.png', cv.IMREAD_GRAYSCALE)
img3 = cv.imread('rio/rio-03.png', cv.IMREAD_GRAYSCALE)
img4 = cv.imread('rio/rio-04.png', cv.IMREAD_GRAYSCALE)


def match(d1, d2):
    n1 = d1.shape[0]
    n2 = d2.shape[0]

    matches = []
    for i in range(n1):
        # 1st cross, i1 is the index in d2 with which the index i in d1 matches
        fv1 = d1[i, :]
        # manhatan distance
        diff = np.abs(d2 - fv1)
        distances = np.sum(diff, axis=1)
        # euclidian distance
        # diff = d2 - fv1
        # diff = np.square(diff)
        # distances = np.sum(diff, axis=1)
        # distances = np.sqrt(distances)
        i1 = np.argmin(distances) # take the index of the min distance, d2 index
        mindist1 = distances[i1] # take the min distance of the i1 index

        # 2nd cross, i is the index in d1 with which the index i1 in d2 matches?
        fv2 = d2[i1, :] # take the index i1 of d2
        # manhatan distance
        diff = np.abs(d1 - fv2)
        distances = np.sum(diff, axis=1)
        # euclid distance
        # diff = d1 - fv2
        # diff = np.square(diff)
        # distances = np.sum(diff, axis=1)
        # distances = np.sqrt(distances)

        i2 = np.argmin(distances)
        mindist2 = distances[i2]

        if i2 == i:
            matches.append(cv.DMatch(i, i1, mindist1))

    return matches

def find_panorama(img1, img2, Sift):
    if sift:
        kp1 = sift.detect(img1)
        desc1 = sift.compute(img1, kp1)
        kp2 = sift.detect(img2)
        desc2 = sift.compute(img2, kp2)
    else:
        kp1 = surf.detect(img1)
        desc1 = surf.compute(img1, kp1)
        kp2 = surf.detect(img2)
        desc2 = surf.compute(img2, kp2)

    # Find matches
    matches = match(desc1[1], desc2[1])

    print(len(matches))

    matches.sort(key=lambda x: x.distance)
    matches = matches[:500] if (len(matches) > 500) else matches
    dimg = cv.drawMatches(img1, desc1[0], img2, desc2[0], matches, None)

    img_pt1 = np.array([kp1[x.queryIdx].pt for x in matches])
    img_pt2 = np.array([kp2[x.trainIdx].pt for x in matches])

    M, mask = cv.findHomography(img_pt2, img_pt1, cv.RANSAC)

    img3 = cv.warpPerspective(img2, M, (img1.shape[1] + 1000, img1.shape[0] + 1000))
    # img3[0: img1.shape[0], 0: img1.shape[1]] = img1

    # cut black lines in the middle
    black_white = np.ones((img1.shape[0], img1.shape[1]), dtype=np.uint8)
    for x in range(img1.shape[0]):
        for y in range(img1.shape[1]):
            if img1[x, y] == 0:
                black_white[x, y] = 0
    kernel = np.ones((5, 5), np.uint8)
    erode = cv.morphologyEx(black_white, cv.MORPH_ERODE, kernel)
    for x in range(erode.shape[0]):
        for y in range(erode.shape[1]):
            if erode[x, y] != 0:
                img3[x, y] = img1[x, y]

    return img3, dimg

def crop(image):
    y_nonzero, x_nonzero = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]


# Save the images
filename = 'rio/sift_rio-01-02-03-04.png'
if not os.path.exists(filename):
    filename = 'rio/sift_rio-01-02.png'
    img_sift1, dimg_sift1 = find_panorama(img1, img2, Sift=True)
    img_sift1 = crop(img_sift1)
    cv.imwrite(filename, img_sift1)
    filename = 'rio/matches_sift_rio-01-02.png'
    cv.imwrite(filename, dimg_sift1)

    filename = 'rio/sift_rio-03-04.png'
    img_sift2, dimg_sift2 = find_panorama(img3, img4, Sift=True)
    img_sift2 = crop(img_sift2)
    cv.imwrite(filename, img_sift2)
    filename = 'rio/matches_sift_rio-03-04.png'
    cv.imwrite(filename, dimg_sift2)

    filename = 'rio/sift_rio-01-02-03-04.png'
    img_sift, dimg_sift = find_panorama(img_sift1, img_sift2, Sift=True)
    img_sift = crop(img_sift)
    cv.imwrite(filename, img_sift)
    filename = 'rio/matches_sift_rio-01-02--03-04.png'
    cv.imwrite(filename, dimg_sift)

filename = 'rio/surf_rio-01-02-03-04.png'
if not os.path.exists(filename):
    filename = 'rio/surf_rio-01-02.png'
    img_surf1, dimg_surf1 = find_panorama(img1, img2, Sift=False)
    img_surf1 = crop(img_surf1)
    cv.imwrite(filename, img_surf1)
    filename = 'rio/matches_surf_rio-01-02.png'
    cv.imwrite(filename, dimg_surf1)

    filename = 'rio/surf_rio-03-04.png'
    img_surf2, dimg_surf2 = find_panorama(img3, img4, Sift=False)
    img_surf2 = crop(img_surf2)
    cv.imwrite(filename, img_surf2)
    filename = 'rio/matches_surf_rio-03-04.png'
    cv.imwrite(filename, dimg_surf2)

    filename = 'rio/surf_rio-01-02-03-04.png'
    img_surf, dimg_surf = find_panorama(img_surf1, img_surf2, Sift=False)
    img_surf = crop(img_surf)
    cv.imwrite(filename, img_surf)
    filename = 'rio/matches_surf_rio-01-02--03-04.png'
    cv.imwrite(filename, dimg_surf)

# Sift
filename = 'rio/sift_rio-01-02.png'
final_img1 = cv.imread(filename, cv.IMREAD_GRAYSCALE)
cv.namedWindow('final1 Sift')
cv.imshow('final1 Sift', final_img1)
cv.waitKey(0)

filename = 'rio/sift_rio-03-04.png'
final_img2 = cv.imread(filename, cv.IMREAD_GRAYSCALE)
cv.namedWindow('final2 Sift')
cv.imshow('final2 Sift', final_img2)
cv.waitKey(0)

filename = 'rio/sift_rio-01-02-03-04.png'
final_img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
cv.namedWindow('final Sift')
cv.imshow('final Sift', final_img)
cv.waitKey(0)

# Surf
filename = 'rio/surf_rio-01-02.png'
final_img1 = cv.imread(filename, cv.IMREAD_GRAYSCALE)
cv.namedWindow('final1 Surf')
cv.imshow('final1 Surf', final_img1)
cv.waitKey(0)

filename = 'rio/surf_rio-03-04.png'
final_img2 = cv.imread(filename, cv.IMREAD_GRAYSCALE)
cv.namedWindow('final2 Surf')
cv.imshow('final2 Surf', final_img2)
cv.waitKey(0)

filename = 'rio/surf_rio-01-02-03-04.png'
final_img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
cv.namedWindow('final Surf')
cv.imshow('final Surf', final_img)
cv.waitKey(0)

# -----------------------------------------------------------------------------------------------------------------

# Read my images
img1 = cv.imread('my_images/img-01.png', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('my_images/img-02.png', cv.IMREAD_GRAYSCALE)
img3 = cv.imread('my_images/img-03.png', cv.IMREAD_GRAYSCALE)
img4 = cv.imread('my_images/img-04.png', cv.IMREAD_GRAYSCALE)

width = int(img1.shape[1] * 0.5)
height = int(img1.shape[0] * 0.5)
dim = (width, height)

# resize images
img1 = cv.resize(img1, dim, interpolation=cv.INTER_AREA)
img2 = cv.resize(img2, dim, interpolation=cv.INTER_AREA)
img3 = cv.resize(img3, dim, interpolation=cv.INTER_AREA)
img4 = cv.resize(img4, dim, interpolation=cv.INTER_AREA)

# Save my images
filename = 'my_images/sift_img-01-02-03-04.png'
if not os.path.exists(filename):
    filename = 'my_images/sift_img-01-02.png'
    img_sift1, dimg_sift1 = find_panorama(img1, img2, Sift=True)
    img_sift1 = crop(img_sift1)
    cv.imwrite(filename, img_sift1)
    filename = 'my_images/matches_sift_img-01-02.png'
    cv.imwrite(filename, dimg_sift1)

    filename = 'my_images/sift_img-03-04.png'
    img_sift2, dimg_sift2 = find_panorama(img3, img4, Sift=True)
    img_sift2 = crop(img_sift2)
    cv.imwrite(filename, img_sift2)
    filename = 'my_images/matches_sift_img-03-04.png'
    cv.imwrite(filename, dimg_sift2)

    filename = 'my_images/sift_img-01-02-03-04.png'
    img_sift, dimg_sift = find_panorama(img_sift1, img_sift2, Sift=True)
    img_sift = crop(img_sift)
    cv.imwrite(filename, img_sift)
    filename = 'my_images/matches_sift_img-01-02--03-04.png'
    cv.imwrite(filename, dimg_sift)

filename = 'my_images/surf_img-01-02-03-04.png'
if not os.path.exists(filename):
    filename = 'my_images/surf_img-01-02.png'
    img_surf1, dimg_surf1 = find_panorama(img1, img2, Sift=False)
    img_surf1 = crop(img_surf1)
    cv.imwrite(filename, img_surf1)
    filename = 'my_images/matches_surf_img-01-02.png'
    cv.imwrite(filename, dimg_surf1)

    filename = 'my_images/surf_img-03-04.png'
    img_surf2, dimg_surf2 = find_panorama(img3, img4, Sift=False)
    img_surf2 = crop(img_surf2)
    cv.imwrite(filename, img_surf2)
    filename = 'my_images/matches_surf_img-03-04.png'
    cv.imwrite(filename, dimg_surf2)

    filename = 'my_images/surf_img-01-02-03-04.png'
    img_surf, dimg_surf = find_panorama(img_surf1, img_surf2, Sift=False)
    img_surf = crop(img_surf)
    cv.imwrite(filename, img_surf)
    filename = 'my_images/matches_surf_img-01-02--03-04.png'
    cv.imwrite(filename, dimg_surf)

# Sift
filename = 'my_images/sift_img-01-02.png'
final_img1 = cv.imread(filename, cv.IMREAD_GRAYSCALE)
cv.namedWindow('my_final1 Sift')
cv.imshow('my_final1 Sift', final_img1)
cv.waitKey(0)

filename = 'my_images/sift_img-03-04.png'
final_img2 = cv.imread(filename, cv.IMREAD_GRAYSCALE)
cv.namedWindow('my_final2 Sift')
cv.imshow('my_final2 Sift', final_img2)
cv.waitKey(0)

filename = 'my_images/sift_img-01-02-03-04.png'
final_img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
cv.namedWindow('my_final Sift')
cv.imshow('my_final Sift', final_img)
cv.waitKey(0)

# Surf
filename = 'my_images/surf_img-01-02.png'
final_img1 = cv.imread(filename, cv.IMREAD_GRAYSCALE)
cv.namedWindow('my_final1 Surf')
cv.imshow('my_final1 Surf', final_img1)
cv.waitKey(0)

filename = 'my_images/surf_img-03-04.png'
final_img2 = cv.imread(filename, cv.IMREAD_GRAYSCALE)
cv.namedWindow('my_final2 Surf')
cv.imshow('my_final2 Surf', final_img2)
cv.waitKey(0)

filename = 'my_images/surf_img-01-02-03-04.png'
final_img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
cv.namedWindow('my_final Surf')
cv.imshow('my_final Surf', final_img)
cv.waitKey(0)


import cv2
import math
import random
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


# Functions
def sift(image1, image2, number_matches=400):
    # Converting image to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Detect features
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Find corresponding points
    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x:x.distance)

    bounch1 = []
    bounch2 = []
    for i in matches[:number_matches]:
        idx1 = i.queryIdx
        idx2 = i.trainIdx
        bounch1.append(np.round(kp1[idx1].pt))
        bounch2.append(np.round(kp2[idx2].pt))
    
    return bounch1, bounch2


def draw_matches(image1, points1, image2, points2):
    # Paste the images side by side
    img12 = np.concatenate((image1, image2), axis=1)
    for i in range(len(points1)):
        ptx1 , pty1 = points1[i]
        ptx2 , pty2 = points2[i]
        ptx2 += image1.shape[1]
        p1 = [ int ( ptx1 ) , int ( pty1 ) ]
        p2 = [ int ( ptx2 ) , int ( pty2 ) ]
        cv2.line( img12, p1, p2, (random.randint(0,255),
                                  random.randint(0,255),
                                  random.randint(0,255)), 1)
    
    img12 = cv2.cvtColor(img12, cv2.COLOR_BGR2RGB)
        
    return img12


def calculate_H(points, points_pr):
    '''
    By Linear Least-Squares Minimization
    Inputs points are in Euclidean form
    '''
    points_num = len(points)

    A = np.zeros( (2*points_num, 8) )
    b = np.zeros( (2*points_num, 1) )
    for i in range(points_num):
        pt    = points[i]
        pt_pr = points_pr[i]

        row1_a = np.array([0, 0, 0, -pt[0], -pt[1], -1, pt_pr[1]*pt[0], pt_pr[1]*pt[1]])
        row2_a = np.array([pt[0], pt[1], 1, 0, 0, 0, -pt_pr[0]*pt[0], -pt_pr[0]*pt[1]])
        A[2*i]   = row1_a
        A[2*i+1] = row2_a

        b[2*i]   = -pt_pr[1]
        b[2*i+1] = pt_pr[0]

    H = np.matmul(np.linalg.pinv(A), b)
    H = np.vstack((H, np.array( [1] )))
    H = H.reshape(3,3)

    return H


def calculate_inliers(points, points_pr, H, threshold):
    '''
    To compare calculated prime-points by using H with correspond one
    and make inlier and outliers list
    Input points are in Euclidean form
    '''
    points_num = len(points)

    # Add 1 to points coordinate to reform them as Homogenious points
    points    = np.insert(points, 2, 1, axis=1)
    points_pr = np.insert(points_pr, 2, 1, axis=1)
    
    # Calculate prime points by using H
    calc_points_pr = np.zeros( (points_num , 3) )
    for i in range(points_num):
        calc_pt_pr = np.dot(H, points[i].T)
        calc_pt_pr /= calc_pt_pr[2]
        calc_points_pr[i] = calc_pt_pr

    img1_inliers  = []
    img2_inliers  = []
    img1_outliers = []
    img2_outliers = []
    for i in range(points_num):
        # Check errors
        error = np.sqrt( (points_pr[i][0] - calc_points_pr[i][0])**2 + \
                         (points_pr[i][1] - calc_points_pr[i][1])**2 )
        if error <= threshold:
            img1_inliers.append(list(points[i]))
            img2_inliers.append(list(points_pr[i]))
        else:
            img1_outliers.append(list(points[i]))
            img2_outliers.append(list(points_pr[i]))

    return img1_inliers, img2_inliers, img1_outliers, img2_outliers


def RANSAC(points, points_pr, error_threshold=3, epsilon=0.35, n=8):
    '''
    RANSAC method to find inliers (noisy data) and remove outliers (false data)
    Input points are in Euclidean form
    '''
    points_num = len(points)
    # RANSAC parameters
    p = 0.99
    N = int(math.log(1-p) / math.log(1-(1-epsilon)**n))
    M = (1 - epsilon) * points_num
    for i in tqdm(range(N)):
        random_indexes = random.sample(range(0, points_num), n)
        points_random = []
        points_pr_random = []
        for idx in random_indexes:
            points_random.append(points[idx])
            points_pr_random.append(points_pr[idx])
        # Check if the point lists have any decimal point
        points_random    = np.round(points_random)
        points_pr_random = np.round(points_pr_random)

        # Calculate H
        H = calculate_H(points_random, points_pr_random)

        # Find inliers and outliers
        img1_inliers, img2_inliers, img1_outliers, img2_outliers = calculate_inliers(points,
                                                                                    points_pr,
                                                                                    H,
                                                                                    error_threshold)

        # If we find intended number of inliers
        if len(img1_inliers) >= M:
            return img1_inliers, img2_inliers, img1_outliers, img2_outliers
    
    return [], [], [], []


def draw_inlier_matches(image1, inliers1, outliers1,
                        image2, inliers2, outliers2):

    # Paste the images side by side
    img12 = np.concatenate((image1, image2), axis=1)
    
    for i in range(len(inliers1)):
        ptx1 , pty1 = inliers1[i][:2]
        ptx2 , pty2 = inliers2[i][:2]
        ptx2 += image1.shape[1]
        p1 = [ int ( ptx1 ) , int ( pty1 ) ]
        p2 = [ int ( ptx2 ) , int ( pty2 ) ]
        cv2.line( img12, p1, p2, (0,255,0), 1)
    
    for i in range(len(outliers1)):
        ptx1 , pty1 = outliers1[i][:2]
        ptx2 , pty2 = outliers2[i][:2]
        ptx2 += image1.shape[1]
        p1 = [ int ( ptx1 ) , int ( pty1 ) ]
        p2 = [ int ( ptx2 ) , int ( pty2 ) ]
        cv2.line( img12, p1, p2, (0,0,255), 1)

    img12 = cv2.cvtColor(img12, cv2.COLOR_BGR2RGB)

    return img12


def create_blank_image(images, H_refference_mid):
    '''
    Make a blank image to use for creating panoramic image (rectification)
    '''
    mid_image_idx = len(images) // 2
    h_mid, w_mid = images[mid_image_idx].shape[:2]
    y_min , x_min , y_max , x_max = 0 , 0 , h_mid , w_mid

    other_images = np.delete(images, mid_image_idx, 0)
    for i in range(len(other_images)):
        
        # Corner coordinates in homogenious form
        H_img = H_refference_mid[i]
        h_img, w_img = images[i].shape[:2]
        boundary_points = [[0,0,1], [0,h_img,1], [w_img,h_img,1], [w_img,0,1]]
        x_list_p = []
        y_list_p = []
        for b_pt in boundary_points:
            point_p = np.dot(H_img, np.array(b_pt).T)
            point_p /= point_p[2]
            x_list_p.append(int(point_p[0]))
            y_list_p.append(int(point_p[1]))
        max_x_p = max(x_list_p)
        min_x_p = min(x_list_p)
        max_y_p = max(y_list_p)
        min_y_p = min(y_list_p)
        
        x_max = max(x_max, max_x_p)
        x_min = min(x_min, min_x_p)
        y_max = max(y_max, max_y_p)
        y_min = min(y_min, min_y_p)

    blank_image = np.zeros( ((y_max - y_min), (x_max - x_min), 3), dtype=np.uint8 )
    
    return blank_image


def generate_panoramic_blank_image(images, all_H):
    '''
    We have five image then the third one is used for H refferencing
    '''
    H_refference_mid = []
    H_13 = np.dot(all_H[1], all_H[0])
    H_refference_mid.append(H_13)
    H_refference_mid.append(all_H[1])
    H_refference_mid.append(np.linalg.inv(all_H[2]))
    H_53 = np.dot(H_refference_mid[2], np.linalg.inv(all_H[3]))
    H_refference_mid.append(H_53)

    # Blank image
    blank_img = create_blank_image(images, H_refference_mid)

    return blank_img, H_refference_mid


def color_estimator(point, image):
    '''
    Estimate the color by bilinear interpolation method
    '''
    estimate_color = [0, 0, 0]
    image_h, image_w = image.shape[:2]

    x_low = math.floor(point[0])
    x_upp = math.ceil(point[0])
    y_low = math.floor(point[1])
    y_upp = math.ceil(point[1])

    if x_upp < image_w and y_upp < image_h:
        w1 = 1 / np.linalg.norm( [point[1]-y_low , point[0]-x_low] )
        w2 = 1 / np.linalg.norm( [point[1]-y_low , point[0]-x_upp] )
        w3 = 1 / np.linalg.norm( [point[1]-y_upp , point[0]-x_low] )
        w4 = 1 / np.linalg.norm( [point[1]-y_upp , point[0]-x_upp] )

        denom = w1 + w2 + w3 + w4

        estimate_color[0] = (w1 * image[y_low][x_low][0] +
                             w2 * image[y_low][x_upp][0] +
                             w3 * image[y_upp][x_low][0] +
                             w4 * image[y_upp][x_upp][0] ) / denom
        
        estimate_color[1] = (w1 * image[y_low][x_low][1] +
                             w2 * image[y_low][x_upp][1] +
                             w3 * image[y_upp][x_low][1] +
                             w4 * image[y_upp][x_upp][1] ) / denom
        
        estimate_color[2] = (w1 * image[y_low][x_low][2] +
                             w2 * image[y_low][x_upp][2] +
                             w3 * image[y_upp][x_low][2] +
                             w4 * image[y_upp][x_upp][2] ) / denom
        
    return estimate_color


def point_in_polygon(point, polygon):
    '''
    Checking if a point is inside a polygon
    '''
    num_vertices = len(polygon)
    x, y = point[0], point[1]
    inside = False
 
    # Store the first point in the polygon and initialize the second point
    p1 = polygon[0]
 
    # Loop through each edge in the polygon
    for i in range(1, num_vertices + 1):
        # Get the next point in the polygon
        p2 = polygon[i % num_vertices]
 
        # Check if the point is above the minimum y coordinate of the edge
        if y > min(p1[1], p2[1]):
            # Check if the point is below the maximum y coordinate of the edge
            if y <= max(p1[1], p2[1]):
                # Check if the point is to the left of the maximum x coordinate of the edge
                if x <= max(p1[0], p2[0]):
                    # Calculate the x-intersection of the line connecting the point to the edge
                    x_intersection = (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
 
                    # Check if the point is on the same line as the edge or to the left of the x-intersection
                    if p1[0] == p2[0] or x <= x_intersection:
                        # Flip the inside flag
                        inside = not inside
 
        # Store the current point as the first point for the next iteration
        p1 = p2
 
    # Return the value of the inside flag
    return inside


def find_prime_point(orig_point, H, min_x_p, min_y_p):
    '''
    Find corresponding point of a point in projected image
    min_x_p : minimum y when calculating projected image
    min_x_p : minimum x when calculating projected image
    '''
    source_point = np.array(orig_point)
    target_point = np.dot(H, source_point.T)
    target_point /= target_point[2]
    i = int(target_point[0] - min_x_p)
    j = int(target_point[1] - min_y_p)

    return [i,j]


# Read images
img1 = cv2.imread("../../images/0.jpg")
img2 = cv2.imread("../../images/1.jpg")
img3 = cv2.imread("../../images/2.jpg")
img4 = cv2.imread("../../images/3.jpg")
img5 = cv2.imread("../../images/4.jpg")
images = [img1, img2, img3, img4, img5]

# Find feature points in each pair
pts12_1, pts12_2 = sift(img1, img2)
pts23_2, pts23_3 = sift(img2, img3)
pts34_3, pts34_4 = sift(img3, img4)
pts45_4, pts45_5 = sift(img4, img5)

# RANSAC for outlier rejection
inliers12_1, inliers12_2, outliers12_1, outliers12_2 = RANSAC(pts12_1, pts12_2)
inliers23_2, inliers23_3, outliers23_2, outliers23_3 = RANSAC(pts23_2, pts23_3)
inliers34_3, inliers34_4, outliers34_3, outliers34_4 = RANSAC(pts34_3, pts34_4)
inliers45_4, inliers45_5, outliers45_4, outliers45_5 = RANSAC(pts45_4, pts45_5)

# Calculate H according to inliers
H_LLS_12 = calculate_H(inliers12_1, inliers12_2)
H_LLS_23 = calculate_H(inliers23_2, inliers23_3)
H_LLS_34 = calculate_H(inliers34_3, inliers34_4)
H_LLS_45 = calculate_H(inliers45_4, inliers45_5)

H_ref = [H_LLS_12, H_LLS_23, H_LLS_34, H_LLS_45]
pano_blank_image, H_ref_to_mid_img = generate_panoramic_blank_image(images, H_ref)

# Rectify all images upon mid image (image 3)
rectified_images_data = []
for k,imagg in enumerate(images):

    if k < 2:
        H_img = H_ref_to_mid_img[k]
    elif k == 2:
        continue
    else:
        H_img = H_ref_to_mid_img[k-1]

    h_img, w_img = imagg.shape[:2]
    boundary_points = [[0,0,1], [0,h_img,1], [w_img,h_img,1], [w_img,0,1]]
    x_list_p = []
    y_list_p = []
    for b_pt in boundary_points:
        point_p = np.dot(H_img, np.array(b_pt).T)
        point_p /= point_p[2]
        x_list_p.append(int(point_p[0]))
        y_list_p.append(int(point_p[1]))
    max_x_p = max(x_list_p)
    min_x_p = min(x_list_p)
    max_y_p = max(y_list_p)
    min_y_p = min(y_list_p)

    blank_image1 = np.zeros( ((max_y_p - min_y_p), (max_x_p - min_x_p), 3), dtype=np.uint8 )

    H, W = blank_image1.shape[:2]
    ROI = [(0, 0), (0, h_img), (w_img, h_img), (w_img, 0)]
    for i in range(W):
        for j in range(H):
            x_orig = i + min_x_p
            y_orig = j + min_y_p

            target_point = np.array([x_orig, y_orig, 1])
            source_point = np.dot(np.linalg.inv(H_img), target_point.T)
            source_point /= source_point[2]

            check_point = (source_point[0], source_point[1])
            if point_in_polygon(check_point, ROI):
                target_color = color_estimator(source_point, imagg)

                blank_image1[j][i] = target_color

    h_image1, w_image1 = blank_image1.shape[:2]
    rectified_images_data.append((blank_image1, h_image1, w_image1, min_x_p, min_y_p))

# Specify all coordinates to paste all sub-masks sid-by-side
ROI_modif = 1
H, W = pano_blank_image.shape[:2]

image1, h_image1, w_image1, min_x1, min_y1 = rectified_images_data[0]
image2, h_image2, w_image2, min_x2, min_y2 = rectified_images_data[1]
image3 = images[2]; h_image3, w_image3 = images[2].shape[:2]
image4, h_image4, w_image4, min_x4, min_y4 = rectified_images_data[2]
image5, h_image5, w_image5, min_x5, min_y5 = rectified_images_data[3]

# Second image
# Boundary points
image1_corners = [(0,0),
                  (0,images[1].shape[0]),
                  (images[1].shape[1],images[1].shape[0]),
                  (images[1].shape[1],0)]
image2_ROI = []
for pt in image1_corners:
    pt_pr = find_prime_point([pt[0], pt[1], 1], H_ref_to_mid_img[1], min_x2, min_y2)
    pt_pr[0] += ROI_modif
    pt_pr[1] += ROI_modif
    image2_ROI.append(pt_pr)

# Find start point for pasting
image12_1_point = find_prime_point(inliers12_1[0], H_ref_to_mid_img[0], min_x1, min_y1)
image12_2_point = find_prime_point(inliers12_2[0], H_ref_to_mid_img[1], min_x2, min_y2)
start_pt_image2_x = int(image12_1_point[0] - image12_2_point[0])
start_pt_image2_y = int(image12_1_point[1] - image12_2_point[1])

# Third image
# Find start point for pasting
image23_2_point = find_prime_point(inliers23_2[0], H_ref_to_mid_img[1], min_x2, min_y2)
image23_3_point = inliers23_3[0]
start_pt_image3_x = int(start_pt_image2_x + image23_2_point[0] - image23_3_point[0])
start_pt_image3_y = int(start_pt_image2_y + image23_2_point[1] - image23_3_point[1])

# Forth image
# Boundary points
image4_corners = [(0,0),
                  (0,images[3].shape[0]),
                  (images[3].shape[1],images[3].shape[0]),
                  (images[3].shape[1],0)]
image4_ROI = []
for pt in image4_corners:
    pt_pr = find_prime_point([pt[0], pt[1], 1], H_ref_to_mid_img[2], min_x4, min_y4)
    pt_pr[0] += ROI_modif
    pt_pr[1] += ROI_modif
    image4_ROI.append(pt_pr)

# Find start point for pasting
image34_3_point = inliers34_3[0]
image34_4_point = find_prime_point(inliers34_4[0], H_ref_to_mid_img[2], min_x4, min_y4)
start_pt_image4_x = int(start_pt_image3_x + image34_3_point[0] - image34_4_point[0])
start_pt_image4_y = int(start_pt_image3_y + image34_3_point[1] - image34_4_point[1])

# Fifth image
# Boundary points
image5_corners = [(0,0),
                  (0,images[4].shape[0]),
                  (images[4].shape[1],images[4].shape[0]),
                  (images[4].shape[1],0)]
image5_ROI = []
for pt in image5_corners:
    pt_pr = find_prime_point([pt[0], pt[1], 1], H_ref_to_mid_img[3], min_x5, min_y5)
    pt_pr[0] += ROI_modif
    pt_pr[1] += ROI_modif
    image5_ROI.append(pt_pr)

# Find start point for pasting
image45_4_point = find_prime_point(inliers45_4[0], H_ref_to_mid_img[2], min_x4, min_y4)
image45_5_point = find_prime_point(inliers45_5[0], H_ref_to_mid_img[3], min_x5, min_y5)
start_pt_image5_x = int(start_pt_image4_x + image45_4_point[0] - image45_5_point[0])
start_pt_image5_y = int(start_pt_image4_y + image45_4_point[1] - image45_5_point[1])

# Create ultimate panoramic image
for i in range(W):
    for j in range(H):

        # image 1
        if i < w_image1 and j < h_image1:
            pano_blank_image[j][i] = image1[j][i]

        # image 2
        if start_pt_image2_x < i < (start_pt_image2_x + w_image2) and \
           start_pt_image2_y < j < (start_pt_image2_y + h_image2):
            check_point = (i-start_pt_image2_x, j-start_pt_image2_y)
            # Ignore black background
            if point_in_polygon(check_point, image2_ROI):
                pano_blank_image[j][i] = image2[j-start_pt_image2_y][i-start_pt_image2_x]

        # image 3
        if start_pt_image3_x < i < (start_pt_image3_x + w_image3) and \
           start_pt_image3_y < j < (start_pt_image3_y + h_image3):
            check_point = (i-start_pt_image3_x, j-start_pt_image3_y)
            # Ignore black background
            # if point_in_polygon(check_point, image2_ROI):
            pano_blank_image[j][i] = image3[j-start_pt_image3_y][i-start_pt_image3_x]

        # image 4
        if start_pt_image4_x < i < (start_pt_image4_x + w_image4) and \
           start_pt_image4_y < j < (start_pt_image4_y + h_image4):
            check_point = (i-start_pt_image4_x, j-start_pt_image4_y)
            # Ignore black background
            if point_in_polygon(check_point, image4_ROI):
                pano_blank_image[j][i] = image4[j-start_pt_image4_y][i-start_pt_image4_x]

        # image 5
        if start_pt_image5_x < i < (start_pt_image5_x + w_image5) and \
           start_pt_image5_y < j < (start_pt_image5_y + h_image5):
            check_point = (i-start_pt_image5_x, j-start_pt_image5_y)
            # Ignore black background
            if point_in_polygon(check_point, image5_ROI):
                pano_blank_image[j][i] = image5[j-start_pt_image5_y][i-start_pt_image5_x]

# Save the result
cv2.imwrite('../../images/results/panorama_result.jpg', pano_blank_image)


import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
 
# Defining the dimensions of checkerboard
CHECKERBOARD = (4,6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
cnt = 0
# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 
 
 
# Defining the world coordinates for 3D points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

prev_img_shape = None
 
# Extracting path of individual image stored in a given directory
images = glob.glob('images/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
     
    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
         
        imgpoints.append(corners2)
 
        # Draw and display the corners
        cnt += 1
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
     
    #cv2.imshow('img',img)
    #cv2.waitKey(500)
 
cv2.destroyAllWindows()
 
h,w = img.shape[:2]
 
"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(cnt)
print("Camera matrix : \n")
print(K)
print("\nradial Distortion coefficients : \n")
print(dist[:2])
print("\nTangential Distortion coefficients : \n")
print(dist[2:])
print("\nRotation vectors : \n")
print(rvecs)
print("\nTranslation vectors : \n")
print(tvecs)

np.savez('calib.npz', mtx=K, dist=dist, rvecs=rvecs, tvecs=tvecs)

## Undistorting image
img = cv2.imread('images/WIN_20230407_18_17_31_Pro.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))

# undistort
dst = cv2.undistort(img, K, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

# cv2.imshow('Undistorted Image', dst)
# cv2.imshow('distorted image', img)
# cv2.waitKey(0)
cv2.imwrite('calibresult.png', dst)

tot_error = 0
n = len(objpoints)
errors = np.zeros((n))
for i in range(n):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error
    errors[i] = error

plt.bar(range(n), errors)
plt.xlabel('Image Index')
plt.ylabel('Re-Projection Error')
plt.title('Re-Projection Error for Each Image')
plt.show()

print( "Mean error: {}".format(np.mean(errors)) )
print( "Std deviation error: {}".format(np.std(errors)) )
i = 0
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
     
    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret:
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, imgpoints[i], ret)
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        img2 = cv2.drawChessboardCorners(img, CHECKERBOARD, imgpoints2, ret)
    
    i+=1
     
    cv2.imshow('image',img)
    cv2.imshow('reprojected image',img2)
    cv2.waitKey(800)
 
cv2.destroyAllWindows()
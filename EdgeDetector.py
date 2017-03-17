'''

EE 4780 - Computer Vision
Homework 2
@author Bryan Stamey
@since 03-10-2017

'''

import cv2
import numpy as np
from scipy import signal

# Step 1
def my_Normalize(img):

    # convert to grayscale
    if len(img.shape) == 3:                         # check if img is color
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # RGB to grayscale
    
    # convert into range of [0,1]
    min_val = np.min(img.ravel())
    max_val = np.max(img.ravel())
    output = (img.astype('float') - min_val) / (max_val - min_val)

    return output
    
# Step 2
def my_DerivativesOfGaussian(img, sigma):
    
    # sobel kernel
    Sx = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1] ]
    Sy = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1] ]

    # gaussian kernel- COPIED FROM EX9
    halfSize = 3 * sigma
    maskSize = 2 * halfSize + 1 
    mat = np.ones((maskSize,maskSize)) / (float)( 2 * np.pi * (sigma**2))
    xyRange = np.arange(-halfSize, halfSize+1)
    xx, yy = np.meshgrid(xyRange, xyRange)    
    x2y2 = (xx**2 + yy**2)    
    exp_part = np.exp(-(x2y2/(2.0*(sigma**2))))
    gSig = mat * exp_part
    
    # convolve to create gx & gy
    gx = signal.convolve2d(Sx, gSig)              # Convolution
    gy = signal.convolve2d(Sy, gSig)
    
    # apply kernels for Ix & Iy
    Ix = cv2.filter2D(img, -1, gx)
    Iy = cv2.filter2D(img, -1, gy)
    
    
    cv2.imshow('normIx', my_Normalize(Ix))
    cv2.imshow('normIy', my_Normalize(Iy))
    
    return Ix, Iy
    
# Step 3
def my_MagAndOrientation(Ix, Iy, t_low):
    
    # compute magnitude
    mag = np.sqrt(Ix**2 + Iy**2)
    
    # normalize magnitude image
    normMag = my_Normalize(mag)
    
    # compute orientation of gradient
    orient = np.arctan2(Iy, Ix)
    
    # round elements of orient
    orientRows = orient.shape[0]
    orientCols = orient.shape[1]

    for i in range(0, orientRows):
        for j in range(0, orientCols):
            if normMag[i,j] > t_low:
                # case 0
                if (orient[i,j] > (- np.pi / 8) and orient[i,j] <= (np.pi / 8)):
                    orient[i,j] = 0
                elif (orient[i,j] > (7 * np.pi / 8) and orient[i,j] <= np.pi):
                    orient[i,j] = 0
                elif (orient[i,j] >= -np.pi and orient[i,j] < (-7 * np.pi / 8)):
                    orient[i,j] = 0
                # case 1
                elif (orient[i,j] > (np.pi / 8) and orient[i,j] <= (3 * np.pi / 8)):
                    orient[i,j] = 3
                elif (orient[i,j] >= (-7 * np.pi / 8) and orient[i,j] < (-5 * np.pi / 8)):
                    orient[i,j] = 3
                # case 2
                elif (orient[i,j] > (3 * np.pi / 8) and orient[i,j] <= (5 * np.pi /8)):
                    orient[i,j] = 2
                elif (orient[i,j] >= (-5 * np.pi / 4) and orient[i,j] < (-3 * np.pi / 8)):
                    orient[i,j] = 2
                # case 3
                elif (orient[i,j] > (5 * np.pi/8) and orient[i,j] <= (7 * np.pi /8)):
                    orient[i,j] = 1
                elif (orient[i,j] >= (-3 * np.pi / 8) and orient[i,j] < (-np.pi / 8)):
                    orient[i,j] = 1

    # convert orientation to color
    orientColor = orient.astype(np.uint8)
    orientColor = cv2.cvtColor(orientColor, cv2.COLOR_GRAY2BGR)
    for i in range(0, orientRows):
        for j in range(0, orientCols):
            if normMag[i,j] > t_low:
                if (orient[i,j] == 0):
                    orientColor[i,j] = [0, 0, 255]
                elif (orient[i,j] == 1):
                    orientColor[i,j] = [0, 255, 0]
                elif (orient[i,j] == 2):
                    orientColor[i,j] = [255, 0, 0]
                elif (orient[i,j] == 3):
                    orientColor[i,j] = [220, 220, 220]

    # show normalized magnitude and orientation images
    cv2.imshow('Normalized Magnitude', normMag)
    cv2.imshow('Rounded Orientation', orientColor)
    
    return normMag, orient
    
# Step 4    
def my_NMS(mag, orient, t_low):
    mag_thin = np.zeros(mag.shape)
    for i in range(mag.shape[0] - 1):
        for j in range(mag.shape[1] - 1):
            if mag[i][j] < t_low:
                continue
            if orient[i][j] == 0:
                if mag[i][j] > mag[i][j-1] and mag[i][j] >= mag[i][j+1]:
                    mag_thin[i][j] = mag[i][j]
            if orient[i][j] == 1:
                if mag[i][j] > mag[i-1][j+1] and mag[i][j] >= mag[i+1][j-1]:
                    mag_thin[i][j] = mag[i][j]
            if orient[i][j] == 2:
                if mag[i][j] > mag[i-1][j] and mag[i][j] >= mag[i+1][j]:
                    mag_thin[i][j] = mag[i][j]
            if orient[i][j] == 3:
                if mag[i][j] > mag[i-1][j-1] and mag[i][j] >= mag[i+1][j+1]:
                    mag_thin[i][j] = mag[i][j]

    cv2.imshow('mag_thin', mag_thin)
    return mag_thin
    
# Step 5
def my_linking(mag_thin, orient, tLow, tHigh):
    result_binary = np.zeros(mag_thin.shape)
    
    # forward scan
    for i in range(0, mag_thin.shape[0] - 1):           # rows
        for j in range(0, mag_thin.shape[1] - 1):       # columns
            if mag_thin[i][j] >= tHigh:
                if mag_thin[i][j+1] >= tLow:            # right
                    mag_thin[i][j+1] = tHigh
                if mag_thin[i+1][j+1] >= tLow:          # bottom right
                    mag_thin[i+1][j+1] = tHigh
                if mag_thin[i+1][j] >= tLow:            # bottom
                    mag_thin[i+1][j] = tHigh
                if mag_thin[i+1][j-1] >= tLow:          # bottom left
                    mag_thin[i+1][j-1] = tHigh
    
    # backwards scan - CHANGED TO -2
    for i in range(mag_thin.shape[0] - 2, 0, -1):       # rows
        for j in range(mag_thin.shape[1] - 2, 0, -1):   # columns
            if mag_thin[i][j] >= tHigh:
                if mag_thin[i][j-1] > tLow:             # left
                    mag_thin[i][j-1] = tHigh
                if mag_thin[i-1][j-1]:                  # top left
                    mag_thin[i-1][j-1] = tHigh
                if mag_thin[i-1][j] > tLow:             # top
                    mag_thin[i-1][j] = tHigh
                if mag_thin[i-1][j+1] > tLow:           # top right
                    mag_thin[i-1][j+1] = tHigh

    # fill in result_binary
    for i in range(0, mag_thin.shape[0] - 1):           # rows
        for j in range(0, mag_thin.shape[1] - 1):       # columns
            if mag_thin[i][j] >= tHigh:
                result_binary[i][j] = 1                 # set to 1 for >= tHigh
                
    return result_binary
    
# Step 6    
def my_Canny(img, sigma, tLow, tHigh):
    img = cv2.imread(img)
    imgNorm = my_Normalize(img)
    cv2.imshow('Normalized Grayscale', imgNorm)
    Ix, Iy = my_DerivativesOfGaussian(imgNorm, 1)
    mag, orient = my_MagAndOrientation(Ix, Iy, tLow)
    mag_thin = my_NMS(mag, orient, tLow)
    result_binary = my_linking(mag_thin, orient, tLow, tHigh)
    cv2.imshow('result_binary', result_binary)
    

###############################################################################   

 
############ Set Values #############
imgName = 'TestImg1.jpg'            #   <- insert image name here
sigma = .2                          #   <- insert sigma value
tLow = 0.01                         #   <- insert tLow value
tHigh = 0.2                         #   <- insert tHigh value
#####################################


# run the Canny edge detector
my_Canny(imgName, sigma, tLow, tHigh)


# wait for esc to terminate
key = cv2.waitKey(0)
key = chr(key & 255)
if key == 27:
    cv2.destroyAllWindows

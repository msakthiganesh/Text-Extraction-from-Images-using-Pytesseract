from PIL import Image
import pytesseract
import argparse
import cv2
import os
import numpy as np
import imutils
from scipy import ndimage
import math
import cld2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
	help="type of preprocessing to be done")
args = vars(ap.parse_args())

#Load the image. 
full_image = cv2.imread(args["image"])

#Shrinking the size of the input image if resolution greater than 1230*692 -----------------------------------
height = full_image.shape[0]
width = full_image.shape[1]
print("height:", height, "width:", width)

while((full_image.shape[0]>= 692) or (full_image.shape[1] >= 1230)):
    full_image = cv2.resize(full_image,None,fx=0.9,fy=0.9,interpolation=cv2.INTER_CUBIC)
    print("Shrunk once here------------------->")
    print("Current Resolution:", full_image.shape[0], "X", full_image.shape[1])
filename = "resized_img.jpg".format(os.getpid())
cv2.imwrite(filename, full_image)

# Shrinking ends here -------------------------------------------------------------------------------------------

#Cropping the required portion from the entire image
# Select ROI
r = cv2.selectROI(full_image)
    
# Crop image
image = full_image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

# Display cropped image
cv2.imshow("Cropped Image", image)
filename = "cropped_img.jpg".format(os.getpid())
cv2.imwrite(filename, image)

#Using hough transform to rectify the orientation of the image
img_before = cv2.imread('cropped_img.jpg')

img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=10, maxLineGap=5)

angles = []

for x1, y1, x2, y2 in lines[0]:
    # cv2.line(img_before, (x1, y1), (x2, y2), (255, 0, 0), 3)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    angles.append(angle)

median_angle = np.median(angles)
image = ndimage.rotate(img_before, median_angle)

print ("Angle is {}".format(median_angle))
cv2.imwrite('hough.jpg', image) 

#Creating a backup 180 degree rotated image
hough180 = imutils.rotate(image, 180)
cv2.imwrite('hough180.jpg', hough180)
# --------DESKEWING IMAGE STARTS HERE-------------------------------------------------------------------------------------------
# convert the image to grayscale and flip the foreground and background to ensure foreground is now "white" and the background is "black"
# flipped_image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# flipped_image=cv2.bitwise_not(flipped_image)

#Using Opening Filter (Erosion followed by Dilation) to remove noise
# kernel = np.ones((5,5),np.uint8)
# flipped_image = cv2.morphologyEx(flipped_image, cv2.MORPH_OPEN, kernel)

#Using Closing Filter(Dilation follwed by erosion) to remove noise
# flipped_image = cv2.morphologyEx(flipped_image, cv2.MORPH_CLOSE, kernel)

#Threshold the image, set all foreground pixels to 255 and all background pixels to 0
# thresh = cv2.threshold(flipped_image,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

#Grab the (x,y) coordinates of all pixel values that are greater than zero, 
#then use these coordinates to compute a rotated bounding box that contains all the coordinates

# coords = np.column_stack(np.where(thresh>0))
# angle = cv2.minAreaRect(coords)[-1]

#The 'cv2.minAreaRect' function returns values in the range [-90,0];
#as the rectangle rotates clockwise the returned angle trends to 0
#in this special case,we need to add 90 degrees to the angle
# if angle <-45:
#     angle = -(90+angle)
#Otherwise, just take the inverse of the angle to make it positive
# else:
#     angle = -angle

#Rotate the image to deskew it
# (h,w) = image.shape[:2]
# center = (w//2, h//2)
# M = cv2.getRotationMatrix2D(center, angle, 1.0)
# image = cv2.warpAffine(image, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

#Draw the correction angle on the image
# cv2.putText(image, "Angle: {:.2f} degrees".format(angle), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

#Show the deskewed image
# print("[INFO] angle: {:.3f}".format(angle))
# cv2.imshow("Deskewed Image", image)

# -------DESKEWING IMAGE ENDS HERE---------------------------------------------------------------------------------------------------

# -------FUNCTION FOR MODIFYING BRIGHTNESS AND CONTRAST STARTS HERE ----------------------------------------------------------------
def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf
# -------FUNCTION FOR MODIFYING BRIGHTNESS AND CONTRAST ENDS HERE----------------------------------------------------------------------

# -------BRIGHTNESS AND CONTRAST MODIFICATIONS STARTS HERE -----------------------------------------------------------------------------

brightcont_image = apply_brightness_contrast(image, 15, 30)
brightcont_image180 = apply_brightness_contrast(hough180, 15, 30)
# cv2.imshow("Brightened/Contrasted Image:", brightcont_image)

# brightcont_image = image
# -------BRIGHTNESS AND CONTRAST MODIFICATIONS ENDS HERE--------------------------------------------------------------------------------


#Convert deskewed/input image to gray scale
gray = cv2.cvtColor(brightcont_image, cv2.COLOR_BGR2GRAY)
gray180 = cv2.cvtColor(brightcont_image180, cv2.COLOR_BGR2GRAY)

# -------FILTERS STARTS HERE-----------------------------------------------------------------------------------------------------------

# flipped_image=cv2.bitwise_not(gray)
# cv2.imshow("Flipped Image:", flipped_image)




# #Threshold the image, set all foreground pixels to 255 and all background pixels to 0
# thresh = cv2.threshold(flipped_image,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# thresh = cv2.adaptiveThreshold(flipped_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)


#Using Binary Thresholding 
# ret,gray = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)



#Threshold the image, set all foreground pixels to 0 and all background pixels to 255
# gray = cv2.threshold(gray,100,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

#Adaptive Gaussian Thresholding to remove noise
# gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

#Rescale the image to increase/decrease the size for better accuracy
# gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

#Optional: Applying median blurring to the image to remve noise
# gray = cv2.medianBlur(gray,3)
# gray = cv2.blur(gray, (2, 2))
# gray = cv2.bilateralFilter(gray,9,75,75)

# -------FILTERS ENDS HERE-----------------------------------------------------------------------------------------------------------

#Write the grayscaled image to the disk as a temporary file so we can apply OCR to it
filename = "final_processed_img.jpg".format(os.getpid())
cv2.imwrite(filename, gray)

filename180 = "final_processed_img180.jpg".format(os.getpid())
cv2.imwrite(filename180, gray180)

#Load the image as PIL/Pillow image, apply OCR, adn then delete the temporary file
text = pytesseract.image_to_string(Image.open(filename), config='--psm 12')

text180 = pytesseract.image_to_string(Image.open(filename180), config='--psm 12')


isReliable, textBytesFound, details = cld2.detect(text)
# print('  details: %s' % str(details))
# print("Details 0 of text:", details[0][3])

isReliable2, textBytesFound2, details2 = cld2.detect(text180)
# print("Details 2 of text:", details2[0][3])
# print('  details: %s' % str(details2))


if(details[0][3] >= details2[0][3]):
    print(text)
    cv2.imshow("Final Output Image:", gray)
else:
    print(text180)
    cv2.imshow("Final Output Image180:", gray180)

cv2.waitKey(0)
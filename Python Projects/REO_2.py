print('=-'*78)
print(' '*60, 'REO 02 - LIST OF EXERCISE')
print('=-'*78)
print('COURSE: VISÃO COMPUTACIONAL NO MELHORAMENTO DE PLANTAS')
print('PROFESSOR: VINÍCIUS QUINTÃO CARNEIRO')
print('STUDENT: ERIC VINICIUS VIEIRA SILVA')
print('DATE: 08/03/2020')
print('=-'*78)
print(' ')

print('-'*50)
print('Packages: numpy, math, matplotlib')
import numpy as np
import cv2
import matplotlib.pyplot as plt
print('-'*50)
print(' ')

print('EXERCISE 01:')
print(' ')
print('Select an image, and using OPENCV package perform the following commands:')
print('1.a) Show the image, and its number of rows, columns, channels and pixels:')
print('Answer:')
soy_image_bgr = cv2.imread('image3.jpg', 1)
soy_image_rgb = cv2.cvtColor(soy_image_bgr, cv2.COLOR_BGR2RGB)
plt.figure("soy_image")
plt.imshow(soy_image_rgb)
plt.show()
lin, col, channels = np.shape(soy_image_rgb)
print("This image has: " + str(lin) + " lines by " + str(col) + " columns; " + str(channels) +
      " channels; and a total of " + str(lin*col) + " pixels.")
print('-'*50)
print(" ")

print('1.b) Cut the image in order to obtain only its region of interest, and then, answer the next questions:')
print('Answer:')
soy_image_cut = soy_image_rgb[70:1330, 140:740, 0:3]  # rows; columns; channels
plt.figure("soy_image_cut")
plt.imshow(soy_image_cut)
plt.show()
print('-'*50)
print(" ")

print('1.c) Convert the image to a grayscale (intensity), then show it as both "Grayscale" and "JET" color maps')
print('Answer:')
soy_image_gray = cv2.cvtColor(soy_image_cut, cv2.COLOR_RGB2GRAY)

plt.figure("soy_image_gray")
plt.subplot(1, 2, 1)
plt.imshow(soy_image_gray, cmap='gray')
plt.title("Grayscale")
plt.xticks([])
plt.yticks([])

plt.subplot(1, 2, 2)
plt.imshow(soy_image_gray, cmap='jet')
plt.title("JET")
plt.xticks([])
plt.yticks([])
plt.show()
print('-'*50)
print(" ")

print('1.d) Show the grayscale image and its histogram; relate them both.')
print('Answer:')
soy_hist = cv2.calcHist([soy_image_gray], [0], None, [256], [0, 256])
plt.figure("GrayScale Histogram")
plt.subplot(2, 1, 1)
plt.imshow(soy_image_gray, cmap="gray")
plt.title("Image")
plt.xticks([])
plt.yticks([])

plt.subplot(2, 1, 2)
plt.plot(soy_hist, color="black")
plt.title("Histogram")
plt.xlim([0, 256])
plt.xlabel("Pixel value")
plt.ylabel("Number of Pixels")
plt.show()

print('-'*50)
print(" ")

print('1.e) Using the GrayScale image, perform image segmentation, in order to remove its background, by using'
      'manual threshold and Otsu technique. Plot the segmented image and its histogram, as well as the final'
      ' color image after segmentation. Explain the results.')
print('Answer:')

manual_threshold = 190

(L1, img_manual_threshold) = cv2.threshold(soy_image_gray, manual_threshold, 255, cv2.THRESH_BINARY)
(L2, img_manual_threshold_inv) = cv2.threshold(soy_image_gray, manual_threshold, 255, cv2.THRESH_BINARY_INV)
(L3, img_otsu_threshold) = cv2.threshold(soy_image_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(L4, img_otsu_threshold_inv) = cv2.threshold(soy_image_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
print("Manual threshold: " + str(L1))
print("Manual threshold_INV: " + str(L2))
print("Otsu threshold: " + str(L3))
print("Otsu threshold_INV: " + str(L3))

soy_hist = cv2.calcHist([soy_image_gray], [0], None, [256], [0, 256])
manual_hist = cv2.calcHist([img_manual_threshold], [0], None, [256], [0, 256])
manual_inv_hist = cv2.calcHist([img_manual_threshold_inv], [0], None, [256], [0, 256])
otsu_hist = cv2.calcHist([img_otsu_threshold], [0], None, [256], [0, 256])
otsu_inv_hist = cv2.calcHist([img_otsu_threshold_inv], [0], None, [256], [0, 256])

soy_seg1 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=img_manual_threshold)
soy_seg2 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=img_manual_threshold_inv)
soy_seg3 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=img_otsu_threshold)
soy_seg4 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=img_otsu_threshold_inv)

plt.figure("Images")
plt.subplot(5, 3, 1)
plt.imshow(soy_image_gray, cmap="gray")
plt.ylabel("GrayScale")
plt.xticks([])
plt.yticks([])

plt.subplot(5, 3, 2)
plt.plot(soy_hist, color="black")
plt.title("Histograms")
plt.xlim([0, 256])
plt.xticks([])
plt.ylabel("N.° of Pixels")

plt.subplot(5, 3, 3)
plt.imshow(soy_image_cut)
plt.ylabel("Original")
plt.xticks([])
plt.yticks([])

plt.subplot(5, 3, 4)
plt.imshow(img_manual_threshold, cmap="gray")
plt.ylabel("Manual")
plt.xticks([])
plt.yticks([])

plt.subplot(5, 3, 5)
plt.plot(manual_hist, color="black")
plt.axvline(x=L1, color='r')
plt.xlim([0, 256])
plt.xticks([])
plt.ylabel("N.° of Pixels")

plt.subplot(5, 3, 6)
plt.imshow(soy_seg1)
plt.ylabel("Manual")
plt.xticks([])
plt.yticks([])

plt.subplot(5, 3, 7)
plt.imshow(img_manual_threshold_inv, cmap="gray")
plt.ylabel("Manual_inv")
plt.xticks([])
plt.yticks([])

plt.subplot(5, 3, 8)
plt.plot(manual_inv_hist, color="black")
plt.axvline(x=L2, color='r')
plt.xlim([0, 256])
plt.xticks([])
plt.ylabel("N.° of Pixels")

plt.subplot(5, 3, 9)
plt.imshow(soy_seg2)
plt.ylabel("Manual_inv")
plt.xticks([])
plt.yticks([])

plt.subplot(5, 3, 10)
plt.imshow(img_otsu_threshold, cmap="gray")
plt.ylabel("Otsu")
plt.xticks([])
plt.yticks([])

plt.subplot(5, 3, 11)
plt.plot(otsu_hist, color="black")
plt.axvline(x=L3, color='r')
plt.xlim([0, 256])
plt.xticks([])
plt.ylabel("N.° of Pixels")

plt.subplot(5, 3, 12)
plt.imshow(soy_seg3)
plt.ylabel("Otsu")
plt.xticks([])
plt.yticks([])

plt.subplot(5, 3, 13)
plt.imshow(img_otsu_threshold_inv, cmap="gray")
plt.ylabel("Otsu_inv")
plt.xticks([])
plt.yticks([])

plt.subplot(5, 3, 14)
plt.plot(otsu_inv_hist, color="black")
plt.axvline(x=L4, color='r')
plt.xlim([0, 256])
plt.xlabel("Pixel value")
plt.ylabel("N.° of Pixels")

plt.subplot(5, 3, 15)
plt.imshow(soy_seg4)
plt.ylabel("Otsu_inv")
plt.xticks([])
plt.yticks([])

plt.show()

print('-'*50)
print(" ")

print('1.f) Plot a figure showing the image in RGB, Lab, HSB and YCrCb color systems')
print('Answer:')

soy_lab = cv2.cvtColor(soy_image_cut, cv2.COLOR_RGB2Lab)
soy_hsv = cv2.cvtColor(soy_image_cut, cv2.COLOR_RGB2HSV)
soy_ycrcb = cv2.cvtColor(soy_image_cut, cv2.COLOR_RGB2YCR_CB)

plt.figure("Color systems")
plt.subplot(1, 4, 1)
plt.imshow(soy_image_cut)
plt.title("RGB")
plt.xticks([])
plt.yticks([])

plt.subplot(1, 4, 2)
plt.imshow(soy_lab)
plt.title("Lab")
plt.xticks([])
plt.yticks([])

plt.subplot(1, 4, 3)
plt.imshow(soy_hsv)
plt.title("HSV")
plt.xticks([])
plt.yticks([])

plt.subplot(1, 4, 4)
plt.imshow(soy_ycrcb)
plt.title("YCRCB")
plt.xticks([])
plt.yticks([])

plt.show()

print('-'*50)
print(" ")

print('1.f) Plot a figure for each color system (RGB, HSV, Lab and YCrCb), '
      'considering each channel and their histograms')
print('Answer:')
print('RGB: channels and histograms')

hist_R = cv2.calcHist([soy_image_cut], [0], None, [256], [0, 256])  # [0] = R
hist_G = cv2.calcHist([soy_image_cut], [1], None, [256], [0, 256])  # [1] = G
hist_B = cv2.calcHist([soy_image_cut], [2], None, [256], [0, 256])  # [2] = B

plt.figure("RGB")
plt.subplot(3, 3, 1)
plt.imshow(soy_image_cut[:, :, 0], cmap="gray")  # All pixels, channel 0
plt.title("R")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 3, 2)
plt.imshow(soy_image_cut[:, :, 1], cmap="gray")  # All pixels, channel 1
plt.title("G")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 3, 3)
plt.imshow(soy_image_cut[:, :, 2], cmap="gray")  # All pixels, channel 2
plt.title("B")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 3, 4)
plt.plot(hist_R, color="r")
plt.title("Histogram")
plt.xlim([0, 256])
plt.xlabel("Pixel value")
plt.ylabel("N.° of Pixels")

plt.subplot(3, 3, 5)
plt.plot(hist_G, color="g")
plt.title("Histogram")
plt.xlim([0, 256])
plt.xlabel("Pixel value")

plt.subplot(3, 3, 6)
plt.plot(hist_B, color="b")
plt.title("Histogram")
plt.xlim([0, 256])
plt.xlabel("Pixel value")

plt.show()

print(" ")

print('HSV: channels and histograms')
hist_H = cv2.calcHist([soy_hsv], [0], None, [256], [0, 256])  # [0] = H
hist_S = cv2.calcHist([soy_hsv], [1], None, [256], [0, 256])  # [1] = S
hist_V = cv2.calcHist([soy_hsv], [2], None, [256], [0, 256])  # [2] = V

plt.figure("HSV")
plt.subplot(3, 3, 1)
plt.imshow(soy_hsv[:, :, 0], cmap="gray")  # All pixels, channel 0
plt.title("H")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 3, 2)
plt.imshow(soy_hsv[:, :, 1], cmap="gray")  # All pixels, channel 1
plt.title("S")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 3, 3)
plt.imshow(soy_hsv[:, :, 2], cmap="gray")  # All pixels, channel 2
plt.title("V")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 3, 4)
plt.plot(hist_H, color="black")
plt.title("Histogram")
plt.xlim([0, 256])
plt.xlabel("Pixel value")
plt.ylabel("N.° of Pixels")

plt.subplot(3, 3, 5)
plt.plot(hist_S, color="black")
plt.title("Histogram")
plt.xlim([0, 256])
plt.xlabel("Pixel value")

plt.subplot(3, 3, 6)
plt.plot(hist_V, color="black")
plt.title("Histogram")
plt.xlim([0, 256])
plt.xlabel("Pixel value")

plt.show()

print(" ")

print('Lab: channels and histograms')
hist_L = cv2.calcHist([soy_lab], [0], None, [256], [0, 256])  # [0] = H
hist_a = cv2.calcHist([soy_lab], [1], None, [256], [0, 256])  # [1] = S
hist_b = cv2.calcHist([soy_lab], [2], None, [256], [0, 256])  # [2] = V

plt.figure("Lab")
plt.subplot(3, 3, 1)
plt.imshow(soy_lab[:, :, 0], cmap="gray")  # All pixels, channel 0
plt.title("L")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 3, 2)
plt.imshow(soy_lab[:, :, 1], cmap="gray")  # All pixels, channel 1
plt.title("a")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 3, 3)
plt.imshow(soy_lab[:, :, 2], cmap="gray")  # All pixels, channel 2
plt.title("b")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 3, 4)
plt.plot(hist_L, color="black")
plt.title("Histogram")
plt.xlim([0, 256])
plt.xlabel("Pixel value")
plt.ylabel("N.° of Pixels")

plt.subplot(3, 3, 5)
plt.plot(hist_a, color="black")
plt.title("Histogram")
plt.xlim([0, 256])
plt.xlabel("Pixel value")

plt.subplot(3, 3, 6)
plt.plot(hist_b, color="black")
plt.title("Histogram")
plt.xlim([0, 256])
plt.xlabel("Pixel value")

plt.show()

print(" ")

print('YCrCb: channels and histograms')
hist_Y = cv2.calcHist([soy_ycrcb], [0], None, [256], [0,256])  # [0] = H
hist_Cr = cv2.calcHist([soy_ycrcb], [1], None, [256], [0,256])  # [1] = S
hist_Cb = cv2.calcHist([soy_ycrcb], [2], None, [256], [0,256])  # [2] = V

plt.figure("YCrCb")
plt.subplot(3, 3, 1)
plt.imshow(soy_ycrcb[:, :, 0], cmap="gray")  # All pixels, channel 0
plt.title("Y")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 3, 2)
plt.imshow(soy_ycrcb[:, :, 1], cmap="gray")  # All pixels, channel 1
plt.title("Cr")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 3, 3)
plt.imshow(soy_ycrcb[:, :, 2], cmap="gray")  # All pixels, channel 2
plt.title("Cb")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 3, 4)
plt.plot(hist_Y, color="black")
plt.title("Histogram")
plt.xlim([0, 256])
plt.xlabel("Pixel value")
plt.ylabel("N.° of Pixels")

plt.subplot(3, 3, 5)
plt.plot(hist_Cr, color="black")
plt.title("Histogram")
plt.xlim([0, 256])
plt.xlabel("Pixel value")

plt.subplot(3, 3, 6)
plt.plot(hist_Cb, color="black")
plt.title("Histogram")
plt.xlim([0, 256])
plt.xlabel("Pixel value")

plt.show()

print('-'*50)
print(" ")

print('1.g) Find the best system color and its channel that gets the best image segmentation result.'
      'Remove image background by using manual and Otsu thresholds. Plot the binary image, its histogram, and'
      'the final color image after segmentation')
print('Answer:')

print("RGB")
manual_threshold_R = 173  # Otsu = 173
manual_threshold_G = 177  # Otsu = 177
manual_threshold_B = 138  # Otsu = 138

# R
(LR1, R_MANUAL) = cv2.threshold(soy_image_cut[:, :, 0], manual_threshold_R, 255, cv2.THRESH_BINARY)
(LR2, R_MANUAL_INV) = cv2.threshold(soy_image_cut[:, :, 0], manual_threshold_R, 255, cv2.THRESH_BINARY_INV)
(LR3, R_OTSU) = cv2.threshold(soy_image_cut[:, :, 0], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(LR4, R_OTSU_INV) = cv2.threshold(soy_image_cut[:, :, 0], 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

soy_R1 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=R_MANUAL)
soy_R2 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=R_MANUAL_INV)
soy_R3 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=R_OTSU)
soy_R4 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=R_OTSU_INV)

# G
(LG1, G_MANUAL) = cv2.threshold(soy_image_cut[:, :, 1], manual_threshold_G, 255, cv2.THRESH_BINARY)
(LG2, G_MANUAL_INV) = cv2.threshold(soy_image_cut[:, :, 1], manual_threshold_G, 255, cv2.THRESH_BINARY_INV)
(LG3, G_OTSU) = cv2.threshold(soy_image_cut[:, :, 1], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(LG4, G_OTSU_INV) = cv2.threshold(soy_image_cut[:, :, 1], 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

soy_G1 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=G_MANUAL)
soy_G2 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=G_MANUAL_INV)
soy_G3 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=G_OTSU)
soy_G4 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=G_OTSU_INV)

# B
(LB1, B_MANUAL) = cv2.threshold(soy_image_cut[:, :, 2], manual_threshold_B, 255, cv2.THRESH_BINARY)
(LB2, B_MANUAL_INV) = cv2.threshold(soy_image_cut[:, :, 2], manual_threshold_B, 255, cv2.THRESH_BINARY_INV)
(LB3, B_OTSU) = cv2.threshold(soy_image_cut[:, :, 2], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(LB4, B_OTSU_INV) = cv2.threshold(soy_image_cut[:, :, 2], 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

soy_B1 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=B_MANUAL)
soy_B2 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=B_MANUAL_INV)
soy_B3 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=B_OTSU)
soy_B4 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=B_OTSU_INV)

plt.figure("RGB")
plt.subplot(3, 5, 1)
plt.imshow(soy_image_cut[:, :, 0], cmap="gray")
plt.title("R")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 2)
plt.imshow(soy_R1)
plt.title("Manual")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 3)
plt.imshow(soy_R2)
plt.title("Manual Inv")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 4)
plt.imshow(soy_R3)
plt.title("Otsu")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 5)
plt.imshow(soy_R4)
plt.title("Otsu Inv")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 6)
plt.imshow(soy_image_cut[:, :, 1], cmap="gray")
plt.title("G")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 7)
plt.imshow(soy_G1)
plt.title("Manual")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 8)
plt.imshow(soy_G2)
plt.title("Manual Inv")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 9)
plt.imshow(soy_G3)
plt.title("Otsu")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 10)
plt.imshow(soy_G4)
plt.title("Otsu Inv")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 11)
plt.imshow(soy_image_cut[:, :, 2], cmap="gray")
plt.title("B")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 12)
plt.imshow(soy_B1)
plt.title("Manual")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 13)
plt.imshow(soy_B2)
plt.title("Manual Inv")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 14)
plt.imshow(soy_B3)
plt.title("Otsu")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 15)
plt.imshow(soy_B4)
plt.title("Otsu Inv")
plt.xticks([])
plt.yticks([])

plt.show()

print(" ")

print("HSV")
manual_threshold_H = 65  # Otsu = 65
manual_threshold_S = 113  # Otsu = 113
manual_threshold_V = 187  # Otsu = 187

# H
(LH1, H_MANUAL) = cv2.threshold(soy_hsv[:, :, 0], manual_threshold_H, 255, cv2.THRESH_BINARY)
(LH2, H_MANUAL_INV) = cv2.threshold(soy_hsv[:, :, 0], manual_threshold_H, 255, cv2.THRESH_BINARY_INV)
(LH3, H_OTSU) = cv2.threshold(soy_hsv[:, :, 0], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(LH4, H_OTSU_INV) = cv2.threshold(soy_hsv[:, :, 0], 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

soy_H1 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=H_MANUAL)
soy_H2 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=H_MANUAL_INV)
soy_H3 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=H_OTSU)
soy_H4 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=H_OTSU_INV)

# S
(LS1, S_MANUAL) = cv2.threshold(soy_hsv[:, :, 1], manual_threshold_S, 255, cv2.THRESH_BINARY)
(LS2, S_MANUAL_INV) = cv2.threshold(soy_hsv[:, :, 1], manual_threshold_S, 255, cv2.THRESH_BINARY_INV)
(LS3, S_OTSU) = cv2.threshold(soy_hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(LS4, S_OTSU_INV) = cv2.threshold(soy_hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

soy_S1 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=S_MANUAL)
soy_S2 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=S_MANUAL_INV)
soy_S3 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=S_OTSU)
soy_S4 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=S_OTSU_INV)

# V
(LV1, V_MANUAL) = cv2.threshold(soy_hsv[:, :, 2], manual_threshold_V, 255, cv2.THRESH_BINARY)
(LV2, V_MANUAL_INV) = cv2.threshold(soy_hsv[:, :, 2], manual_threshold_V, 255, cv2.THRESH_BINARY_INV)
(LV3, V_OTSU) = cv2.threshold(soy_hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(LV4, V_OTSU_INV) = cv2.threshold(soy_hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

soy_V1 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=V_MANUAL)
soy_V2 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=V_MANUAL_INV)
soy_V3 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=V_OTSU)
soy_V4 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=V_OTSU_INV)

plt.figure("HSV")
plt.subplot(3, 5, 1)
plt.imshow(soy_hsv[:, :, 0], cmap="gray")
plt.title("H")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 2)
plt.imshow(soy_H1)
plt.title("Manual")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 3)
plt.imshow(soy_H2)
plt.title("Manual Inv")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 4)
plt.imshow(soy_H3)
plt.title("Otsu")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 5)
plt.imshow(soy_H4)
plt.title("Otsu Inv")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 6)
plt.imshow(soy_hsv[:, :, 1], cmap="gray")
plt.title("S")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 7)
plt.imshow(soy_S1)
plt.title("Manual")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 8)
plt.imshow(soy_S2)
plt.title("Manual Inv")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 9)
plt.imshow(soy_S3)
plt.title("Otsu")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 10)
plt.imshow(soy_S4)
plt.title("Otsu Inv")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 11)
plt.imshow(soy_hsv[:, :, 2], cmap="gray")
plt.title("V")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 12)
plt.imshow(soy_V1)
plt.title("Manual")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 13)
plt.imshow(soy_V2)
plt.title("Manual Inv")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 14)
plt.imshow(soy_V3)
plt.title("Otsu")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 15)
plt.imshow(soy_V4)
plt.title("Otsu Inv")
plt.xticks([])
plt.yticks([])

plt.show()

print(" ")

print("Lab")
manual_threshold_L = 182  # Otsu = 182
manual_threshold_a = 120  # Otsu = 120
manual_threshold_b = 154  # Otsu = 154

# L
(LL1, L_MANUAL) = cv2.threshold(soy_lab[:, :, 0], manual_threshold_L, 255, cv2.THRESH_BINARY)
(LL2, L_MANUAL_INV) = cv2.threshold(soy_lab[:, :, 0], manual_threshold_L, 255, cv2.THRESH_BINARY_INV)
(LL3, L_OTSU) = cv2.threshold(soy_lab[:, :, 0], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(LL4, L_OTSU_INV) = cv2.threshold(soy_lab[:, :, 0], 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

soy_L1 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=L_MANUAL)
soy_L2 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=L_MANUAL_INV)
soy_L3 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=L_OTSU)
soy_L4 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=L_OTSU_INV)

# a
(La1, a_MANUAL) = cv2.threshold(soy_lab[:, :, 1], manual_threshold_a, 255, cv2.THRESH_BINARY)
(La2, a_MANUAL_INV) = cv2.threshold(soy_lab[:, :, 1], manual_threshold_a, 255, cv2.THRESH_BINARY_INV)
(La3, a_OTSU) = cv2.threshold(soy_lab[:, :, 1], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(La4, a_OTSU_INV) = cv2.threshold(soy_lab[:, :, 1], 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

soy_a1 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=a_MANUAL)
soy_a2 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=a_MANUAL_INV)
soy_a3 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=a_OTSU)
soy_a4 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=a_OTSU_INV)

# b
(Lb1, b_MANUAL) = cv2.threshold(soy_lab[:, :, 2], manual_threshold_b, 255, cv2.THRESH_BINARY)
(Lb2, b_MANUAL_INV) = cv2.threshold(soy_lab[:, :, 2], manual_threshold_b, 255, cv2.THRESH_BINARY_INV)
(Lb3, b_OTSU) = cv2.threshold(soy_lab[:, :, 2], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(Lb4, b_OTSU_INV) = cv2.threshold(soy_lab[:, :, 2], 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

soy_b1 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=b_MANUAL)
soy_b2 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=b_MANUAL_INV)
soy_b3 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=b_OTSU)
soy_b4 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=b_OTSU_INV)

plt.figure("Lab")
plt.subplot(3, 5, 1)
plt.imshow(soy_lab[:, :, 0], cmap="gray")
plt.title("L")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 2)
plt.imshow(soy_L1)
plt.title("Manual")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 3)
plt.imshow(soy_L2)
plt.title("Manual Inv")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 4)
plt.imshow(soy_L3)
plt.title("Otsu")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 5)
plt.imshow(soy_L4)
plt.title("Otsu Inv")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 6)
plt.imshow(soy_lab[:, :, 1], cmap="gray")
plt.title("a")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 7)
plt.imshow(soy_a1)
plt.title("Manual")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 8)
plt.imshow(soy_a2)
plt.title("Manual Inv")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 9)
plt.imshow(soy_a3)
plt.title("Otsu")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 10)
plt.imshow(soy_a4)
plt.title("Otsu Inv")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 11)
plt.imshow(soy_lab[:, :, 2], cmap="gray")
plt.title("b")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 12)
plt.imshow(soy_b1)
plt.title("Manual")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 13)
plt.imshow(soy_b2)
plt.title("Manual Inv")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 14)
plt.imshow(soy_b3)
plt.title("Otsu")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 15)
plt.imshow(soy_b4)
plt.title("Otsu Inv")
plt.xticks([])
plt.yticks([])

plt.show()

print(" ")

print("YCrCb")
manual_threshold_Y = 174  # Otsu = 174
manual_threshold_Cr = 136  # Otsu = 136
manual_threshold_Cb = 99  # Otsu = 99

# Y
(LY1, Y_MANUAL) = cv2.threshold(soy_ycrcb[:, :, 0], manual_threshold_Y, 255, cv2.THRESH_BINARY)
(LY2, Y_MANUAL_INV) = cv2.threshold(soy_ycrcb[:, :, 0], manual_threshold_Y, 255, cv2.THRESH_BINARY_INV)
(LY3, Y_OTSU) = cv2.threshold(soy_ycrcb[:, :, 0], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(LY4, Y_OTSU_INV) = cv2.threshold(soy_ycrcb[:, :, 0], 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

soy_Y1 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=Y_MANUAL)
soy_Y2 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=Y_MANUAL_INV)
soy_Y3 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=Y_OTSU)
soy_Y4 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=Y_OTSU_INV)

# Cr
(LCr1, Cr_MANUAL) = cv2.threshold(soy_ycrcb[:, :, 1], manual_threshold_Cr, 255, cv2.THRESH_BINARY)
(LCr2, Cr_MANUAL_INV) = cv2.threshold(soy_ycrcb[:, :, 1], manual_threshold_Cr, 255, cv2.THRESH_BINARY_INV)
(LCr3, Cr_OTSU) = cv2.threshold(soy_ycrcb[:, :, 1], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(LCr4, Cr_OTSU_INV) = cv2.threshold(soy_ycrcb[:, :, 1], 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

soy_Cr1 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=Cr_MANUAL)
soy_Cr2 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=Cr_MANUAL_INV)
soy_Cr3 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=Cr_OTSU)
soy_Cr4 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=Cr_OTSU_INV)

# b
(LCb1, Cb_MANUAL) = cv2.threshold(soy_ycrcb[:, :, 2], manual_threshold_Cb, 255, cv2.THRESH_BINARY)
(LCb2, Cb_MANUAL_INV) = cv2.threshold(soy_ycrcb[:, :, 2], manual_threshold_Cb, 255, cv2.THRESH_BINARY_INV)
(LCb3, Cb_OTSU) = cv2.threshold(soy_ycrcb[:, :, 2], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(LCb4, Cb_OTSU_INV) = cv2.threshold(soy_ycrcb[:, :, 2], 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

soy_Cb1 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=Cb_MANUAL)
soy_Cb2 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=Cb_MANUAL_INV)
soy_Cb3 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=Cb_OTSU)
soy_Cb4 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=Cb_OTSU_INV)

plt.figure("YCrCb")
plt.subplot(3, 5, 1)
plt.imshow(soy_ycrcb[:, :, 0], cmap="gray")
plt.title("Y")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 2)
plt.imshow(soy_Y1)
plt.title("Manual")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 3)
plt.imshow(soy_Y2)
plt.title("Manual Inv")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 4)
plt.imshow(soy_Y3)
plt.title("Otsu")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 5)
plt.imshow(soy_Y4)
plt.title("Otsu Inv")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 6)
plt.imshow(soy_ycrcb[:, :, 1], cmap="gray")
plt.title("Cr")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 7)
plt.imshow(soy_Cr1)
plt.title("Manual")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 8)
plt.imshow(soy_Cr2)
plt.title("Manual Inv")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 9)
plt.imshow(soy_Cr3)
plt.title("Otsu")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 10)
plt.imshow(soy_Cr4)
plt.title("Otsu Inv")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 11)
plt.imshow(soy_ycrcb[:, :, 2], cmap="gray")
plt.title("Cb")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 12)
plt.imshow(soy_Cb1)
plt.title("Manual")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 13)
plt.imshow(soy_Cb2)
plt.title("Manual Inv")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 14)
plt.imshow(soy_Cb3)
plt.title("Otsu")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 15)
plt.imshow(soy_Cb4)
plt.title("Otsu Inv")
plt.xticks([])
plt.yticks([])

plt.show()

print("Best fit was observed on channel S of HSV system")

plt.figure("RESULT")
plt.subplot(2, 4, 4)
plt.imshow(soy_image_cut, cmap="gray")
plt.xticks([])
plt.yticks([])
plt.ylabel("Original")

plt.subplot(2, 4, 8)
plt.imshow(soy_image_cut, cmap="gray")
plt.xticks([])
plt.yticks([])
plt.ylabel("Original")

plt.subplot(2, 4, 1)
plt.imshow(S_MANUAL, cmap="gray")
plt.xticks([])
plt.yticks([])
plt.ylabel("Manual Threshold")


plt.subplot(2, 4, 5)
plt.imshow(S_OTSU, cmap="gray")
plt.xticks([])
plt.yticks([])
plt.ylabel("Otsu Threshold")

plt.subplot(2, 4, 2)
plt.plot(hist_S, color="black")
plt.axvline(x=LS1, color="r")
plt.title("Histogram")
plt.xlim([0, 256])
plt.xticks([])
plt.ylabel("N.° of Pixels")

plt.subplot(2, 4, 6)
plt.plot(hist_S, color="black")
plt.axvline(x=LS3, color="r")
plt.xlim([0, 256])
plt.xlabel("Pixel value")
plt.ylabel("N.° of Pixels")

plt.subplot(2, 4, 3)
plt.imshow(soy_S1)
plt.ylabel("Manual")
plt.xticks([])
plt.yticks([])

plt.subplot(2, 4, 7)
plt.imshow(soy_S3)
plt.ylabel("Otsu")
plt.xticks([])
plt.yticks([])

plt.show()

print('-'*50)
print(' ')

print('1.i) Using the image from (h) as mask, obtain the histograms for each RGB channel:')
print('Answer:')

new_hist_R = cv2.calcHist([soy_image_cut], [0], S_OTSU, [256], [0, 256])
new_hist_G = cv2.calcHist([soy_image_cut], [1], S_OTSU, [256], [0, 256])
new_hist_B = cv2.calcHist([soy_image_cut], [2], S_OTSU, [256], [0, 256])

plt.figure("Mask histograms")
plt.subplot(3, 1, 1)
plt.plot(new_hist_R, color="r")
plt.title("R")
plt.xlim([0, 256])
plt.xticks([])
plt.ylabel("N.° of Pixels")

plt.subplot(3, 1, 2)
plt.plot(new_hist_G, color="g")
plt.title("G")
plt.xlim([0, 256])
plt.xticks([])
plt.ylabel("N.° of Pixels")

plt.subplot(3, 1, 3)
plt.plot(new_hist_B, color="b")
plt.title("B")
plt.xlim([0, 256])
plt.xlabel("Pixel value")
plt.ylabel("N.° of Pixels")

plt.show()

print('-'*50)
print(' ')

print('1.j) Perform arithmetic operations on RGB image in order to highlight the aspect of interest:')
print('Answer:')

R, G, B = cv2.split(soy_image_cut)
# new_soy = (2*G-R-B)/(2*G+R+B)
# new_soy = 0.2989*R + 0.5870*G + 0.1140*B
# new_soy = (2*R-0.5*G)
# new_soy = (G-R)/(G+R)
# new_soy = (G-R)\(G+R-B)
# new_soy = 0.299*R + 0.587*G + 0.114*B  # Y (YCrCb)
# new_soy = -0.168*R - 0.331*G + 0.500*B  # Cr (YCrCb)
# new_soy = 0.500*R - 0.418*G - 0.081*B  #Cb (YCrCb)
# new_soy = 0.4124*R + 0.3576*G + 0.1805*B  # x (XYZ)
new_soy1 = 0.2126*R - 0.7152*G + 0.722*B  # Y  (XYZ)
# new_soy = 0.0193*R - 0.11952*G - 0.9505*B  #  Z (XYZ)
# new_soy = 0.2126*R + 0.7152*G + 0.0722*B  # L (LAB)
new_soy2 = 1.4749*(0.2213*R - 0.3390*G + 0.1177*B) + 128  # A (LAB)
new_soy3 = 0.6245*(0.1949*R + 0.6057*G - 0.8006*B) + 128  # B (LAB)

print("Options 1, 2 and 3 found on: Chaudhary et al. (2012) Color Transformed Based Approach for Disease Spot "
      "Detection on Plant Leaf. International Journal of Computer Science and Telecommunications, 3:6.")

new_soy1 = new_soy1.astype(np.uint8)
hist_new1 = cv2.calcHist([new_soy1], [0], S_OTSU, [256], [0, 256])
new_soy2 = new_soy2.astype(np.uint8)
hist_new2 = cv2.calcHist([new_soy2], [0], S_OTSU, [256], [0, 256])
new_soy3 = new_soy3.astype(np.uint8)
hist_new3 = cv2.calcHist([new_soy3], [0], S_OTSU, [256], [0, 256])

(L1, rec_new_bin_1) = cv2.threshold(new_soy1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(L2, rec_new_bin_1_inv) = cv2.threshold(new_soy1, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
(L3, rec_new_bin_2) = cv2.threshold(new_soy2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(L4, rec_new_bin_2_inv) = cv2.threshold(new_soy2, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
(L5, rec_new_bin_3) = cv2.threshold(new_soy3, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
(L6, rec_new_bin_3_inv) = cv2.threshold(new_soy3, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

rec_seg_Fer_1 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=rec_new_bin_1)
rec_seg_Fer_1_inv = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=rec_new_bin_1_inv)
rec_seg_Fer_2 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=rec_new_bin_2)
rec_seg_Fer_2_inv = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=rec_new_bin_2_inv)
rec_seg_Fer_3 = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=rec_new_bin_3)
rec_seg_Fer_3_inv = cv2.bitwise_and(soy_image_cut, soy_image_cut, mask=rec_new_bin_3_inv)

plt.figure("Chaudhary et al. (2012) Color Transformed Based Approach for Disease Spot Detection on Plant Leaf")
plt.subplot(3, 5, 1)
plt.imshow(soy_image_cut)
plt.ylabel("Original")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 2)
plt.imshow(rec_new_bin_1, cmap="gray")
plt.ylabel("Channel Y (XYZ)")
plt.xticks([])
plt.yticks([])
plt.subplot(3, 5, 3)
plt.imshow(rec_seg_Fer_1)
plt.ylabel("Channel Y (XYZ)")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 4)
plt.imshow(rec_new_bin_1_inv, cmap="gray")
plt.ylabel("Channel Y (XYZ) - INV")
plt.xticks([])
plt.yticks([])
plt.subplot(3, 5, 5)
plt.imshow(rec_seg_Fer_1_inv)
plt.ylabel("Channel Y (XYZ) - INV")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 6)
plt.imshow(soy_image_cut)
plt.ylabel("Original")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 7)
plt.imshow(rec_new_bin_2, cmap="gray")
plt.ylabel("Channel A (LAB)")
plt.xticks([])
plt.yticks([])
plt.subplot(3, 5, 8)
plt.imshow(rec_seg_Fer_2)
plt.ylabel("Channel A (LAB)")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 9)
plt.imshow(rec_new_bin_2_inv, cmap="gray")
plt.ylabel("Channel A (LAB) - INV")
plt.xticks([])
plt.yticks([])
plt.subplot(3, 5, 10)
plt.imshow(rec_seg_Fer_2_inv)
plt.ylabel("Channel A (LAB) - INV")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 11)
plt.imshow(soy_image_cut)
plt.ylabel("Original")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 12)
plt.imshow(rec_new_bin_3, cmap="gray")
plt.ylabel("Channel B (LAB)")
plt.xticks([])
plt.yticks([])
plt.subplot(3, 5, 13)
plt.imshow(rec_seg_Fer_3)
plt.ylabel("Channel B (LAB)")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 5, 14)
plt.imshow(rec_new_bin_3_inv, cmap="gray")
plt.ylabel("Channel B (LAB) - INV")
plt.xticks([])
plt.yticks([])
plt.subplot(3, 5, 15)
plt.imshow(rec_seg_Fer_3_inv)
plt.ylabel("Channel B (LAB) - INV")
plt.xticks([])
plt.yticks([])

plt.show()


print("Image background removed")

rec_seg_Fer_1 = cv2.bitwise_and(soy_S3, soy_S3, mask=rec_new_bin_1)
rec_seg_Fer_1_inv = cv2.bitwise_and(soy_S3, soy_S3,  mask=rec_new_bin_1_inv)
rec_seg_Fer_2 = cv2.bitwise_and(soy_S3, soy_S3,  mask=rec_new_bin_2)
rec_seg_Fer_2_inv = cv2.bitwise_and(soy_S3, soy_S3,  mask=rec_new_bin_2_inv)
rec_seg_Fer_3 = cv2.bitwise_and(soy_S3, soy_S3,  mask=rec_new_bin_3)
rec_seg_Fer_3_inv = cv2.bitwise_and(soy_S3, soy_S3,  mask=rec_new_bin_3_inv)

plt.figure("Image background removed - Chaudhary et al. (2012).")
plt.subplot(3, 6, 1)
plt.imshow(soy_image_cut)
plt.ylabel("Original")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 6, 2)
plt.imshow(soy_S3)
plt.ylabel("Background removed")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 6, 3)
plt.imshow(rec_new_bin_1, cmap="gray")
plt.ylabel("Channel Y (XYZ)")
plt.xticks([])
plt.yticks([])
plt.subplot(3, 6, 4)
plt.imshow(rec_seg_Fer_1)
plt.ylabel("Channel Y (XYZ)")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 6, 5)
plt.imshow(rec_new_bin_1_inv, cmap="gray")
plt.ylabel("Channel Y (XYZ) - INV")
plt.xticks([])
plt.yticks([])
plt.subplot(3, 6, 6)
plt.imshow(rec_seg_Fer_1_inv)
plt.ylabel("Channel Y (XYZ) - INV")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 6, 7)
plt.imshow(soy_image_cut)
plt.ylabel("Original")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 6, 8)
plt.imshow(soy_S3)
plt.ylabel("Background removed")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 6, 9)
plt.imshow(rec_new_bin_2, cmap="gray")
plt.ylabel("Channel A (LAB)")
plt.xticks([])
plt.yticks([])
plt.subplot(3, 6, 10)
plt.imshow(rec_seg_Fer_2)
plt.ylabel("Channel A (LAB)")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 6, 11)
plt.imshow(rec_new_bin_2_inv, cmap="gray")
plt.ylabel("Channel A (LAB) - INV")
plt.xticks([])
plt.yticks([])
plt.subplot(3, 6, 12)
plt.imshow(rec_seg_Fer_2_inv)
plt.ylabel("Channel A (LAB) - INV")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 6, 13)
plt.imshow(soy_image_cut)
plt.ylabel("Original")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 6, 14)
plt.imshow(soy_S3)
plt.ylabel("Background removed")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 6, 15)
plt.imshow(rec_new_bin_3, cmap="gray")
plt.ylabel("Channel B (LAB)")
plt.xticks([])
plt.yticks([])
plt.subplot(3, 6, 16)
plt.imshow(rec_seg_Fer_3)
plt.ylabel("Channel B (LAB)")
plt.xticks([])
plt.yticks([])

plt.subplot(3, 6, 17)
plt.imshow(rec_new_bin_3_inv, cmap="gray")
plt.ylabel("Channel B (LAB) - INV")
plt.xticks([])
plt.yticks([])
plt.subplot(3, 6, 18)
plt.imshow(rec_seg_Fer_3_inv)
plt.ylabel("Channel B (LAB) - INV")
plt.xticks([])
plt.yticks([])

plt.show()

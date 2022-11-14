import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_coordinates(lane_image, line_parameters):
    slope, intercept = line_parameters
    y1 = lane_image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array((x1, y1, x2, y2))


def averge_slope_intercept(lane_image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(lane_image, left_fit_average)
    right_line = make_coordinates(lane_image, right_fit_average)
    return np.array([left_line, right_line])


# Make a function for Canny
def cannyFun(lane_image):
    # convert color image to grayscale
    gray_image = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)

    # Reduce the noise and smoothness of image by using gaussian blur.
    reduce_noise_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply Canny Algorithm
    canny = cv2.Canny(reduce_noise_image, 50, 150)

    return canny


# find the area by making polygon
def regionArea(lane_image):
    height = lane_image.shape[0]
    polygons = np.array([
        [
            (200, height),
            (1100, height),
            (550, 250)
        ]
    ])
    mask = np.zeros_like(lane_image)
    cv2.fillPoly(mask, polygons, 255)
    # Using bitwise operation we croped the area
    maskedImage = cv2.bitwise_and(lane_image, mask)
    return maskedImage


def displayLines(lane_image, lines):
    lineImage = np.zeros_like(lane_image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(lineImage, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lineImage


# read the image from our file and return in multidimensional numpy array containing relative intensity of each pixel.
# image = cv2.imread("test_image.jpg")

# copy of original image in new variable
# lane_image = np.copy(image)
#
# canny = cannyFun(lane_image)
#
# #Croped area image after bitwise operation
# cropedImage = regionArea(canny)
#
# #Find the straight line
# lines = cv2.HoughLinesP(cropedImage,2,np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=5)
#
# average_line = averge_slope_intercept(lane_image,lines)
# lineImage = displayLines(lane_image,average_line)
# comboImage = cv2.addWeighted(lane_image,0.8,lineImage,1,1)


# imshow() takes to argument first is name of the window and second argument which image we want to show.
# cv2.imshow("result", comboImage)
# plt.imshow(canny)
# waitKey() fun is used to wait the image while showing
# cv2.waitKey(0)
# plt.show()


# VIDEO

cap = cv2.VideoCapture("test2.mp4")
while (cap.isOpened()):
    _, frame = cap.read()
    canny = cannyFun(frame)
    cropedImage = regionArea(canny)
    lines = cv2.HoughLinesP(cropedImage, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    average_line = averge_slope_intercept(frame, lines)
    lineImage = displayLines(frame, average_line)
    comboImage = cv2.addWeighted(frame, 0.8, lineImage, 1, 1)
    cv2.imshow("result", comboImage)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

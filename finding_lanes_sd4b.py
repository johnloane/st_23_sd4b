import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    #image = cv2.imread("road.jpg")
    cap = cv2.VideoCapture("JohnM50.mp4")
    while(cap.isOpened()):
        _, frame = cap.read()
        if frame is not None:
            frame = cv2.resize(frame, (960, 1280))
        lane_image = np.copy(frame)
        canny_image = canny(lane_image)
        cropped_image = region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_image, 1, np.pi/180, 100, np.array([]), 
                                minLineLength=40, maxLineGap=5)
        averaged_lines = average_slope_intercept(lane_image, lines)
        line_image = display_lines(lane_image, averaged_lines)
        combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
        cv2.imshow("Combo image", combo_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if abs(slope) < 0.001:
            continue
        elif slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (9/20))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])






def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return line_image





def region_of_interest(image):
    height = image.shape[0]
    triangle = np.array([
        [(0, 700), (500, 300), (960, 1000)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_image = cv2.Canny(blur, 50, 100)
    return canny_image


main()
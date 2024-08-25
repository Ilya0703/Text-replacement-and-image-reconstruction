import cv2
import numpy as np

def joinBoxes(image, boxes):
    resultBoxes = []
    heatmap = np.zeros((image.shape[0], image.shape[1]), dtype="uint8")
    for bbox in boxes:
        x_min, y_min, x_max, y_max = bbox
        x_min -= 10
        x_max += 10
        heatmap[y_min:y_max, x_min:x_max] = 1
    heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
    contours = cv2.findContours(heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        resultBoxes.append([x, y, x + w, y + h])
    return resultBoxes

import cv2
import numpy as np

def getTextColorValue(array):
    values, counts = np.unique(array, return_counts=True)
    sortedIndices = np.argsort(counts)[::-1]
    if (sortedIndices.size == 1):
        secondLargestCountIndex = sortedIndices[0]
    else:
        secondLargestCountIndex = sortedIndices[1]
    return int(values[secondLargestCountIndex])

def getTextColor(image, bbox):
    borderPixels = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    blueChannel, greenChannel, redChannel = cv2.split(borderPixels)
    averageColorBlue = getTextColorValue(blueChannel)
    averageColorGreen = getTextColorValue(greenChannel)
    averageColorRed = getTextColorValue(redChannel)
    return (averageColorBlue, averageColorGreen, averageColorRed)
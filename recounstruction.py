import cv2
import numpy as np

RECONSTRUCTION_BORDER_ERROR_PX = 5

def getMostFrequencyValue(array):
    values, counts = np.unique(array, return_counts=True)
    return values[np.argmax(counts)]

def reconstruct(image, bbox):
    h, w, _ = image.shape

    borderPixels = image.copy()
    x_min = max(bbox[0] - 5, 0)
    x_max = min(bbox[2] + 5, image.shape[0])
    y_min = max(bbox[1] - 5, 0)
    y_max = min(bbox[3] + 5, image.shape[1])

    borderPixels = borderPixels[y_min:y_max, x_min:x_max]
    borderPixels[RECONSTRUCTION_BORDER_ERROR_PX:-RECONSTRUCTION_BORDER_ERROR_PX,
    RECONSTRUCTION_BORDER_ERROR_PX:-RECONSTRUCTION_BORDER_ERROR_PX] = [0, 0, 0]
    blueChannel, greenChannel, redChannel = cv2.split(borderPixels)
    averageColorBlue = getMostFrequencyValue(blueChannel[blueChannel != 0])
    averageColorGreen = getMostFrequencyValue(greenChannel[greenChannel != 0])
    averageColorRed = getMostFrequencyValue(redChannel[redChannel != 0])
    averageColor = (int(averageColorBlue),
                    int(averageColorGreen), int(averageColorRed))
    image = cv2.rectangle(image, (bbox[0] - 1, bbox[1] - 1),
                          (bbox[2] + 1, bbox[3] + 1), averageColor, -1)
    return image

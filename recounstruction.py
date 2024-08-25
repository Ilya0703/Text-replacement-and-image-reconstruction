import cv2
import numpy as np

def getMostFrequencyValue(array):
    values, counts = np.unique(array, return_counts=True)
    return values[np.argmax(counts)]

def reconstruct(image, bbox):
    h, w, _ = image.shape

    borderPixels = image.copy()
    borderPixels = borderPixels[bbox[1] - 5:bbox[3] + 5, bbox[0] - 5:bbox[2] + 5]
    borderPixels[5:-5, 5:-5] = [0, 0, 0]
    blueChannel, greenChannel, redChannel = cv2.split(borderPixels)
    averageColorBlue = getMostFrequencyValue(blueChannel[blueChannel != 0])
    averageColorGreen = getMostFrequencyValue(greenChannel[greenChannel != 0])
    averageColorRed = getMostFrequencyValue(redChannel[redChannel != 0])
    averageColor = (int(averageColorBlue),
                    int(averageColorGreen), int(averageColorRed))
    image = cv2.rectangle(image, (bbox[0] - 1, bbox[1] - 1),
                          (bbox[2] + 1, bbox[3] + 1), averageColor, -1)
    cv2.imwrite("after_reconstruction.png", image)
    return image

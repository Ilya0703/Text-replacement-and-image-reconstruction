import pytesseract
import cv2

def recognizeText(image):
    totalBboxes = []
    height, width, _ = image.shape

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresholdImage = cv2.threshold(gray, 140, 255,
                                        1, cv2.THRESH_BINARY)
    text = pytesseract.image_to_boxes(thresholdImage)
    for box in text.splitlines():
        box = box.split()
        x, y, x1, y1 = (int(box[1]), height - int(box[4]),
                        int(box[3]), height - int(box[2]))
        totalBboxes.append([x, y, x1, y1])

    return totalBboxes

import cv2

def getFontSize(text, bbox, font, thickness):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    fontScale = 2
    while True:
        (textWidth, textHeight), _ = cv2.getTextSize(text,
                                                     font, fontScale, thickness)
        if textWidth <= width and textHeight <= height:
            return fontScale, textWidth, textHeight
        fontScale -= 0.05
        if fontScale < 0.1:
            return None

def insertNewText(image, text, bbox):
    font = cv2.FONT_HERSHEY_COMPLEX
    thickness = 1
    fontScale, textWidth, textHeight = getFontSize(text, bbox, font, thickness)
    targetPoint = (bbox[0] + int((bbox[2] - bbox[0] - textWidth) / 2),
                   bbox[1] + int((bbox[3] - bbox[1] - textHeight) / 2))
    cv2.putText(image, text, targetPoint, font, fontScale,
                (0, 0, 0), thickness, cv2.LINE_AA)

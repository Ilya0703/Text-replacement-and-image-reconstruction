import cv2
import text_recognition
import recounstruction
import text_insert
import join_bboxes

image = cv2.imread("img_1.png")

boxes = text_recognition.recognizeText(image)
joinedBoxes = join_bboxes.joinBoxes(image, boxes)
joinedBoxes = sorted(joinedBoxes, key=lambda x: (x[1], x[0]))

imageReconstructed = image.copy()
for box in joinedBoxes:
    imageReconstructed = recounstruction.reconstruct(imageReconstructed, box)
cv2.imwrite("after_reconstruction.png", imageReconstructed)
index = 0
with open("new_text.txt", 'r') as file:
    lines = file.readlines()
    for line in lines:
        if line[-1] == '\n':
            line = line[:-1]
        text_insert.insertNewText(imageReconstructed, line,
                                  joinedBoxes[index])
        index += 1

cv2.imwrite("result.png", imageReconstructed)

import numpy as np
import sys
import cv2

net = cv2.dnn.readNet('yolov3_training_18200.weights', 'yolov3_testing.cfg')

location = sys.argv[1]
splitted = location.split('\\')
name = splitted[len(splitted) - 1].split('.')[0]

with open("classes.txt", "r") as f:
    classes = f.read().splitlines()

img = cv2.imread(location)

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))


height, width, channels = img.shape

blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)
output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)

boxes = []
confidences = []
class_ids = []

for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)
            x = int(center_x - w/2)
            y = int(center_y - h/2)

            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

if len(indexes) > 0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i] * 100)) + '%'
        color = colors[i]
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, label + " " + confidence, (x + 5, y + 25), font, 1.5, (255, 255, 255), 2)

detected = []

for index in indexes:
    detected.append(classes[class_ids[index]])


if len(detected) == 0:
    print("None", end='', sep='')
else:
    i = 0
    for object in detected:
        if i != len(detected) - 1:
            print(object + ";", end='', sep='')
        else:
            print(object, end='', sep='')
        i = i + 1


cv2.imwrite(name + '_output.png', img)


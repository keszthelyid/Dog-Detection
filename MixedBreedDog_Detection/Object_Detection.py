import numpy as np
from Prediction import Prediction
import sys
import cv2

net = cv2.dnn.readNetFromDarknet('yolov3_testing_mixed.cfg', 'yolov3_training_18200.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


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
predictions = []
good_predictions = []

for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:
            pred = Prediction()
            i = 0
            for num in scores:
                if num > 0.1:
                    pred.dogtype.append(classes[i])
                    pred.percentage.append(round(num * 100, 2))
                i += 1

            predictions.append(pred)

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

for idx in indexes:
   good_predictions.append(predictions[idx])

if len(indexes) > 0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        color = colors[i]
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)


detected = []

for index in indexes:
    detected.append(classes[class_ids[index]])


if len(detected) == 0:
    print("None", end='', sep='')
else:
    i = 0
    for obj in good_predictions:
        if i != len(good_predictions) - 1:
            for index in range(len(good_predictions[i].dogtype)):
                print(obj.dogtype[index] + ":" + str(obj.percentage[index]), end='', sep='')
                if len(obj.dogtype) - 1 != index:
                    print("*", end='', sep='')
            print(";", end='', sep='')
        else:
            for index in range(len(good_predictions[i].dogtype)):
                print(obj.dogtype[index] + ":" + str(obj.percentage[index]), end='', sep='')
                if len(obj.dogtype) - 1 != index:
                    print("*", end='', sep='')
        i = i + 1


cv2.imwrite(name + '_output.png', img)

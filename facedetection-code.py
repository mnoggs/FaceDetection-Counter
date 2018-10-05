import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import json
import time
from pathlib2 import Path

DATA_FILENAME = 'data.json'

ts = time.time()

# create the file if it doesnt already exist
data_file = Path("./" + DATA_FILENAME)
if not data_file.is_file():
    with open(DATA_FILENAME, mode='w') as f:
        json.dump({'people': []}, f)

# detection code
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

camera = cv2.VideoCapture(0)
count = 0

while 1:
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects, wei = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    for c in rects:
        (x, y, w, h) = c
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Diff", frame)
    key = cv2.waitKey(1) & 0xFF

    if time.time() > ts + 180:
        break

    if key == ord('q'):
        break

count = len(rects)



# open file and store json in data list
with open(DATA_FILENAME, mode='r') as f:
    data = json.load(f)
    f.close()

# append new data to the list or overwrite the list
if len(data['people']) == 9:
    data['people'] = [count]
else:
    data['people'].append(count)

# save all the data to the file
with open(DATA_FILENAME, mode='w') as f:
    json.dump(data, f)
    f.close()

cv2.destroyAllWindows()

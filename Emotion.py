from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np


detection_model = 'haarcascade_frontalface_default.xml'
emotion_model = '_mini_XCEPTION.102-0.66.hdf5'

stud_face_detector = cv2.CascadeClassifier(detection_model)
stud_emotion_classifier = load_model(emotion_model, compile=False)
EMOTIONS = ["Angry", "Disgust", "Scared", "Happy", "Sad", "Surprised",
            "Neutral"]


mcamera = cv2.VideoCapture(0)
while True:
    frame = mcamera.read()[1]

    frame = imutils.resize(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = stud_face_detector.detectMultiScale(gray, scaleFactor=1.32, minNeighbors=5)

    canvas = np.zeros((230, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces

        region_of_interest = gray[fY:fY + fH, fX:fX + fW]
        region_of_interest = cv2.resize(region_of_interest, (64, 64))
        region_of_interest = region_of_interest.astype("float") / 255.0
        region_of_interest = img_to_array(region_of_interest)
        region_of_interest = np.expand_dims(region_of_interest, axis=0)

        predictions = stud_emotion_classifier.predict(region_of_interest)[0]
        emotion_probability = np.max(predictions)
        Emotion_label = EMOTIONS[predictions.argmax()]

        for (i, (emotion, probability)) in enumerate(zip(EMOTIONS, predictions)):

            text = "{}: {:.2f}%".format(emotion, probability * 100)

            w = int(probability * 300)
            cv2.rectangle(canvas, (7, (i * 30) + 5), (w, (i * 30) + 30), (0, 255, 0), -1)
            cv2.putText(canvas, text, (10, (i * 30) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
            cv2.putText(frameClone, Emotion_label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 1)

    cv2.imshow('Emotion Capturing...', frameClone)
    cv2.imshow("Emotions in percentage %", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

mcamera.release()
cv2.destroyAllWindows()

import os
import cv2
import numpy as np
from keras.models import load_model, model_from_json
from keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing import image
model = model_from_json(open("../static/model.json", "r").read())
model.load_weights('static/model.h5')
# model = load_model('static\Fer2013.h5')
face_haar_cascade = cv2.CascadeClassifier('../static\haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)

while cap.isOpened():
    res,frame=cap.read()
    frame = cv2.resize(frame,(1080,720))
    height, width , channel = frame.shape
    sub_img = frame[0:int(height/6),0:int(width)]

    black_rect = np.ones(sub_img.shape, dtype=np.uint8)*0
    res = cv2.addWeighted(sub_img, 0.77, black_rect,0.23, 0)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.8
    FONT_THICKNESS = 2
    label_color = (10, 10, 200)
    label = "Emotion Detection System"
    label_dimension = cv2.getTextSize(label,FONT ,FONT_SCALE,FONT_THICKNESS)[0]
    textX = int((res.shape[1] - label_dimension[0]) / 2)
    textY = int((res.shape[0] + label_dimension[1]) / 2)
    cv2.putText(res, label, (textX,textY), FONT, FONT_SCALE, (0,0,0), FONT_THICKNESS)
    gray_image= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_haar_cascade.detectMultiScale(gray_image, minNeighbors=15)
    num_faces = len(faces)
    if num_faces:
        num_faces = num_faces
    else:
        num_faces = 0
    cv2.putText(res, f"Faces Detected: {num_faces}", (5, textY + 30), FONT, 0.5, (0, 255, 0), 1)

    try:
        for (x,y, w, h) in faces:
            cv2.rectangle(frame, pt1 = (x,y),pt2 = (x+w, y+h), color = (255,0,0),thickness =  2)
            roi_gray = gray_image[y-5:y+h+5,x-5:x+w+5]
            roi_gray=cv2.resize(roi_gray,(48,48))
            image_pixels = img_to_array(roi_gray)
            image_pixels = np.expand_dims(image_pixels, axis = 0)
            image_pixels /= 255
            predictions = model.predict(image_pixels)
            max_index = np.argmax(predictions[0])
            emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            emotion_prediction = emotion_detection[max_index]
            label_pred = "Sentiment: {}".format(emotion_prediction)
            cv2.putText(frame, label_pred, (int(x),int(y+h+20)), FONT,0.7, label_color,2)
            label_violation = 'Confidence: {}'.format(str(np.round(np.max(predictions[0])*100,1))+ "%")
            violation_text_dimension = cv2.getTextSize(label_violation,FONT,FONT_SCALE,FONT_THICKNESS )[0]
            violation_y_axis = int((y+h+20) + violation_text_dimension[1])
            cv2.putText(frame, label_violation, (int(x),violation_y_axis+4), FONT,0.7, label_color,2)
    except :
        pass
    frame[0:int(height/6),0:int(width)] =res
    cv2.imshow('frame', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows


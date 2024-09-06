import cv2
import numpy as np
import playsound  # To play sound

def faceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, bboxs

# Load models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb" 

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
agelist = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(43-53)', '(60-100)']
genderList = ['Male', 'Female']

# Start video capture
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    frame, bboxs = faceBox(faceNet, frame)
    
    # Initialize counters
    male_count = 0
    female_count = 0
    
    for bbox in bboxs:
        face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        # Gender Prediction
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]
        
        # Increment gender-specific counters
        if gender == 'Male':
            male_count += 1
        else:
            female_count += 1
        
        # Age Prediction
        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = agelist[agePred[0].argmax()]
        
        # Display label with gender and age
        label = "{},{}".format(gender, age)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 2)
    
    # Display male and female count on the frame
    cv2.putText(frame, f'Males: {male_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f'Females: {female_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Check ratio and play alert sound if male count is significantly higher
    if female_count > 0 and (male_count / female_count) > 4:
        playsound.playsound('alert.mp3')  # Replace 'alert.mp3' with your alert sound file path
    
    # Show the video feed with annotations
    cv2.imshow("Age-Gender-Person Detection", frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
video.release()
cv2.destroyAllWindows()

import cv2
from Face_trainer import Face_trainer as ft
print(cv2.__file__)

cap = cv2.VideoCapture(0)
color = (0,0,255)
stroke = 4

face_cascades = cv2.CascadeClassifier('../cascades/data/haarcascade_frontalface_alt2.xml')

def computer_vision():
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascades.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y), (x+w, y+h),color, stroke)
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    print("quitting computer VS")

def train():
    facetrain = ft(cv = cv2 , cascade = face_cascades)
    facetrain.train()

#Recognices and locates the face and returns a list of tupels with (who, confidence, x_axis, y_axis, width of face, height of face)
def recognice_face(frame):
    face_list = []
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascades.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x,y,w,h) in faces:
        roi = frame[y:y+h, x:x+w]
        person, confidence =  recognizer.recognice(roi)
        face_list.append((person, confidence, x,y,w,h))
    return face_list

#If just the position of the face is important then this is the way to go!
def recogice_position():
    pass


def recognice_face():
    pass
            
if __name__ == "__main__":
    train()
#    cap = cv2.VideoCapture(0)
#    while True:
#        ret, frame = cap.read()
#        print(recognice_face(frame))
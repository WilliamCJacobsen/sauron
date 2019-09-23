import cv2
print(cv2.__file__)

cap = cv2.VideoCapture(0)
color = (0,0,255)
stroke = 4

face_cascades = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

class CV:
    def __init__():


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


if __name__ == "__main__":
    computer_vision()
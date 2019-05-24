import cv2, os

SQUARE_COLOR = (255, 0, 0)
SQUARE_SIZE = 2

xml_path = 'glasses.xml'
clf = cv2.CascadeClassifier(xml_path)
cap = cv2.VideoCapture(0)

cap.set(3, 300)
cap.set(4, 300)
print('W: ' +str(cap.get(3)))
print('H: ' +str(cap.get(4)))

while(not cv2.waitKey(20) & 0xFF == ord('q')):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        recognize  = clf.detectMultiScale(gray)
        for x, y, w, h in recognize:
            cv2.rectangle(frame, (x, y), (x+w, y+h), SQUARE_COLOR, SQUARE_SIZE)
        cv2.imshow('frame',frame)

cap.release()
cv2.destroyAllWindows()

import cv2

detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
imp_img = cv2.VideoCapture("soccer.png")

res, img = imp_img.read()

#convert the image into grayscale, the detector is designed for grayscale 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#returns the (x, y, width, height) of the face
faces = detect.detectMultiScale(gray, 1.3, 5)


for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 255, 0), 2)


cv2.imshow("Image", img)
cv2.waitKey(0)
imp_img.release()
cv2.destroyAllWindows()
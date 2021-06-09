import cv2, glob

detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

images = glob.glob("*.jpg") + glob.glob("*.png") + glob.glob("*.JPG")




for image in images:
    img = cv2.imread(image)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect.detectMultiScale(gray_img, 1.3, 5)

    for (x, y, w, h) in faces:
        final_img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Face", final_img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    

import cv2,time

def photo1(face_cascade,smile_cascade):
    face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    smile_cascade=cv2.CascadeClassifier("smile.xml")
    #photo1(face_cascade,smile_cascade)
    photo = '2.jpg'
    original_image = cv2.imread(photo)
    if original_image is not None:
        image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(image=image, scaleFactor=1.3, minNeighbors=4)
        #profiles_not_faces = [x for x in detected_smiles if x not in detected_faces]
        #print(profiles_not_faces)
        for (x, y, width, height) in detected_faces:
            cv2.rectangle(original_image,(x, y),(x + width, y + height),(0,255,0),thickness=2)
            #detected_smiles = smile_cascade.detectMultiScale(image=image, scaleFactor=1.3, minNeighbors=4)
            smile=smile_cascade.detectMultiScale(image,scaleFactor=1.8,minNeighbors=50)
            for x1,y1,w,h in smile:
                img=cv2.rectangle(original_image,(x1,y1),(x1+w,y1+h),(255,0,0),thickness=10)

        cv2.imshow(f'Detected Faces in {photo}', original_image)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
    else:
        print(f'En error occurred while trying to load {photo}')
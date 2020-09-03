#Haar Algo used for obj,face detection based on viola jones algo
#Find face
#Crop Face
#Detect Smile
#Put the image background back


import cv2

#Face classifier
#Pre trained model for detecting faces
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

#Webcam feed
webcam=cv2.VideoCapture(0)

#To read single frame
"""
#Read webcam feed
frame_read, frame = webcam.read()  #return tuple #1st element bool

#Show image to screen
cv2.imshow("Smile Detector",frame)

#Wait until key is pressed
cv2.waitKey()
"""

#To read multiple frame
while True:
    #Read current frame from webcam video stream
    successful_frame_read, frame = webcam.read()

    #If error, abort
    if not successful_frame_read:
        break

    #Change to Grayscale #optimizes color image
    #cvtColor -> convertcolor
    frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #Detect faces
    #Tell us where(x,y coordinates) faces are in the image (array of rectangles(points) depend on number of person)
    faces=face_detector.detectMultiScale(frame_grayscale) 

    #Detect smiles
    #scalefactor=1.7 how much you blur the image #optimization over greay image
    #minNeighbor=20 neighboring rectangle to actually be counted as smile
    smiles=smile_detector.detectMultiScale(frame_grayscale,1.7,20)

    #Run face detection within each of the faces
    for (x,y,w,h) in faces:

        #Draw rectangle around the face
        #(x,y)->Top left point
        #(x+w,y+h)->Bottom right point
        #(B,G,R)->(100,200,50) A green color
        # 4 is thickness of box
        cv2.rectangle(frame,(x,y),(x+w,y+h),(100,200,50),4)

        #Get sub frame (N-dimensional array slicing)
        the_face=frame[y:y+h,x:x+w]
        
        #Convert RGB face to greyscale
        face_grayscale = cv2.cvtColor(the_face,cv2.COLOR_BGR2GRAY)

        #Detect smiles
        smiles=smile_detector.detectMultiScale(face_grayscale,1.7,20)
        
        #find all smiles in face
        """for(x_,y_,w_,h_) in smiles:

            #Draw rectangle around the smile
            cv2.rectangle(the_face,(x_,y_),(x_+w_,y_+h_),(50,50,200),4)
        """

        if(len(smiles)>0):
            
            #placing smiling position by adding 40 to y-cordinate
            #fontscale is size
            #fontface is FONT_HERSHEY_PLAIN
            #color is white
            cv2.putText(frame,'smiling',(x,y+h+40),fontScale=3,
                        fontFace=cv2.FONT_HERSHEY_PLAIN,color=(255,255,255))
            
    #Show current frame
    cv2.imshow("Smile Detector",frame)

    #Refresh every 1 second
    cv2.waitKey(1) 
    
#Cleanup
webcam.release()
cv2.destroyAllWindows()

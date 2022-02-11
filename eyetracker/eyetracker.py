# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 23:51:44 2022

@author: anirudh
"""

import cv2

import numpy as np

import mouse

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

def move_mouse(x,y):
      
        


    

def contour(eye):
    rows, cols = eye.shape
    eye_2 = eye
    eye_2 = cv2.GaussianBlur(eye_2,(5,5),0)


    _, threshold = cv2.threshold(eye_2, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=False)
    

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        cv2.drawContours(eye_2, [cnt], -1, (0, 0, 255), 3)
        cv2.rectangle(eye, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(eye, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(eye, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
        move_mouse(x,y)
        break
    
        cv2.imshow("eye",eye_2)
        
        
    



while cap.isOpened():
    _, img = cap.read()
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    gray = np.array(gray, dtype='uint8')
    
    face = face_cascade.detectMultiScale(gray,1.1,4) #face detection
    
    for (x,y,w,h) in face:
        
        
        cv2.rectangle(gray,(x,y) , (x+w,y+h), (255,0,0), 2)
        
        roi = gray[y:y+h,x:x+w] #face
        
        eye = eye_cascade.detectMultiScale(roi) #eye in face
        
        
        
        for (a,b,c,d) in eye :
            
            print(a,b)
            
            cv2.rectangle(roi,(a,b) , (a+c,b+d), (0,255,0), 2)
            
            if a-b>50:
                
               z = roi[b:b+d,a:a+c] # z = eye
               
               
               contour(z)
               
        cv2.imshow("face",roi)
            

            

                            



    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

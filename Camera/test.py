import boto3
import json
import time
import requests
import cv2
import os
import numpy as np
import uuid

s3 = boto3.resource('s3')
reko = boto3.client('rekognition')
camera_addr="/dev/video0"
#camera_addr="rtsp://admin:MAMG940901EN0@10.32.89.135"
#camera_addr='0'
cascPath = 'haar-cascade.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
factor = 1
def scene (boxes, actors, frame_size_x,frame_size_y, min_closeness=0.2):
    actors_uid = actors['id']
    actors = actors['pos']
    new_boxes = []
    new_actors = []
    try:
        n_a = actors.shape[0]
    except:
        n_a=0
    try:
        n_b = boxes.shape[0]
    except:
        n_b=0
    if n_a > 0: #there are actors already
        for x,y,w,h in boxes:
            x1_rel = x / frame_size_x
            y1_rel = y / frame_size_y
            x2_rel = (x + w) / frame_size_x
            y2_rel = (y + h) / frame_size_y
            box = [x1_rel, y1_rel, x2_rel, y2_rel]
            new_boxes.append(box)
        for x,y,w,h in actors:
            x1_rel = x / frame_size_x
            y1_rel = y / frame_size_y
            x2_rel = (x + w) / frame_size_x
            y2_rel = (y + h) / frame_size_y
            box = [x1_rel, y1_rel, x2_rel, y2_rel]
            new_actors.append(box)
        
        distance = np.zeros([n_b,n_a])
        i=0
        for b in new_boxes:
            #check distance with existing actors
            if n_a>0:
                j=0
                for a in new_actors:
                    distance[i,j] = np.linalg.norm(np.subtract(b,a))
                    j+=1
            i+=1
        #distance becomes a matrix with the norm between each box [dim 0] and each actor [dim 1]
        updated_actors = np.zeros([n_a,1],dtype=bool)
        for i in range(n_b): #assign boxes to the closest actor or create new actor if no actor is close enough
            closest_actor_distance = min(distance[i,:])
            closest_actor_index = np.argmin(distance[i,:])
            if closest_actor_distance < min_closeness: #update actor
                updated_actors[closest_actor_index] = 1
                actors[closest_actor_index] = boxes[i]
            else:#create new actor
                actors = np.vstack([actors,boxes[i]])
                updated_actors=np.vstack([updated_actors,1])
                uniqueid = str(uuid.uuid4())
                actors_uid = np.vstack([actors_uid,uniqueid])
        print('actors before filtering' + str(actors_uid))

        actors_temp = np.empty((0,actors.shape[1]), int)
        actors_uid_temp = np.empty((0,1), str)
        for i in range(updated_actors.shape[0]):#keep only the most recent actors
            if updated_actors[i]:
                actors_temp = np.vstack([actors_temp,actors[i]])
                actors_uid_temp = np.vstack([actors_uid_temp,actors_uid[i]])
        actors = actors_temp
        actors_uid = actors_uid_temp
        #print(updated_actors)
    else: #actors are empty, just copy all boxes
        actors=boxes
        try:
            n_a = actors.shape[0]
        except:
            n_a = 0
        if n_a > 0:
            for i in range(n_a):
                uniqueid=str(uuid.uuid4())
                actors_uid=np.vstack([actors_uid,uniqueid])
    actors={
        'id':actors_uid,
        'pos':actors
    }
    return actors

print("Connecting to " + camera_addr + " ...")
vcap = cv2.VideoCapture(camera_addr)
vcap.set(20,1)
vcap.set(5,6)
ret, frame = vcap.read() #hay que leerlo una vez para poder ver sus propiedades
print (frame.shape)
frame_size_x = vcap.get(3) #guarda el tama;o del frame
frame_size_y = vcap.get(4)
print("Succesfully connected :D")

actors={
    'id':np.empty([0,1], dtype = str),
    'pos':np.empty([0,4], dtype = int)
}





while True:
    '''boxes_rand = np.empty([0,4], dtype=int)
    n_b = np.random.randint(0,5)
    print('Generating '+str(n_b)+' random boxes')
    for i in range(n_b):#random number of boxes
        boxes_rand=np.vstack([boxes_rand, np.reshape(np.random.randint(0,400,[4,1]),-1)])
    print (boxes_rand)'''

    ret, frame = vcap.read()
    if ret: #revisa que si se haya recibido una imagen para que no se rompa
        print("vcap worked")
        frame_small = cv2.resize(frame, (int(frame_size_x/factor), int(frame_size_y/factor)))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        boxes = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(5, 5),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        boxes = np.multiply(boxes,factor)
    try:
        n_a = actors['pos'].shape[0]
    except:
        n_a = 0
    if n_a > 0:
        for x,y,h,w in actors['pos']: #there are actors, draw them
            #print(x,y,h,w)
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255, 0), 1)
            a=1
    if isinstance(boxes, list):  
        if boxes.shape[0]>0:
            for x,y,h,w in boxes: #there are boxes, draw them
                #print(x,y,h,w)
                #cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 0, 255), 1)
                a=1

    try:
        cv2.imshow("test",frame_small)
    except:
        pass
    cv2.waitKey(1)
    print('actors[id] before = ' + str(actors['id']))
    actors_new = scene(boxes,actors,frame_size_x,frame_size_y)
    print('actors_new[id] after = ' + str(actors_new['id']))
    
    appeared = {
        'id':np.empty([0,1], dtype = str),
        'pos':np.empty([0,4], dtype = int)
    }
    for i in range(actors_new['id'].shape[0]):
        unique = True
        for j in range(actors['id'].shape[0]):
            #check if it existed already, else, append it to the appeared array
            if(actors_new['id'][i]==actors['id'][j]):
                unique = False
        if unique:
            appeared['id'] = np.vstack([appeared['id'],actors_new['id'][i]])
            appeared['pos'] = np.vstack([appeared['pos'],actors_new['pos'][i]])
            x,y,w,h = actors_new['pos'][i]
            face_crop = frame[y:y+h,x:x+w]
            try:
                cv2.imshow("crop" + str(i),face_crop)
            except:
                pass
    print('appeared: '+ str(appeared['id']))
    
    actors = actors_new


    

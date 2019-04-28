import boto3
import json
import time
import requests
import cv2
import os
from shutil import copyfile
import numpy as np

s3 = boto3.resource('s3')
reko = boto3.client('rekognition')
faces_path = './faces/'
not_processed = './not_processed/'
findings_path = './findings/'

import firebase_admin
from firebase_admin import credentials, firestore, storage, db

cred = credentials.Certificate('crowdlytics-1-firebase-adminsdk-7bj5s-4af8c62b68.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'crowdlytics-1.appspot.com',
    'databaseURL': 'https://crowdlytics-1.firebaseio.com/'
})
fs_client = firestore.client()
doc_ref_image_data = db.reference('/image-data')
bucket = storage.bucket()


while True:
    file_list = []
    file_list = os.listdir(findings_path)
    if len(file_list)>100:
        print("file list too long, moving all current images to not_processed")
        #something happened, move all the files to a not processed folder
        for f in range(len(file_list)):
            image_path = findings_path + file_list[f]
            copyfile(image_path, not_processed + file_list[f])
            os.remove(image_path)
    file_list = os.listdir(findings_path)
    if len(file_list)>0:
        print(file_list)
        creation = []
        for f in file_list:
            stat = (os.stat(findings_path + f))
            creation.append(stat.st_ctime)
        #print(creation)
        faceId = None
        oldest_i = np.argmin(creation)
        image_path = findings_path + file_list[oldest_i]
        with open( image_path , 'rb' ) as myimage:
            image=myimage.read()
        try:
            search=reko.search_faces_by_image(
                CollectionId = 'hackaton',
                Image = {
                    'Bytes': image
                },
                MaxFaces = 1,
                FaceMatchThreshold=95.0
            )
            face_info = reko.detect_faces(
                Image = {
                    'Bytes': image,
                },
                Attributes=["ALL"]
            )
            print(face_info["FaceDetails"][0]["AgeRange"], face_info["FaceDetails"][0]["Gender"])
            if search.get('FaceMatches'):
                faceId=search['FaceMatches'][0]['Face']['FaceId']
                print(faceId)

                #enviar a firebase
                headers = {'Content-type': 'application/json'}
                data = {"faceId":faceId, 'timestamp':time.time() }
                #print (data)
                #r = requests.post("https://valo-hk.firebaseio.com/log.json", headers=headers, data=json.dumps(data))
                doc_ref_image_data.push({
                    "bucketname": "crowdlytics-1.appspot.com",
                    "cam-id": "1",
                    "filename" : file_list[oldest_i],
                    "AgeRange":(face_info["FaceDetails"][0]["AgeRange"]["High"] + face_info["FaceDetails"][0]["AgeRange"]["Low"])/2,
                    "Gender": face_info["FaceDetails"][0]["Gender"]["Value"],
                    "timestamp": time.time()
                })
                #print (r)
            else:
                print('detected face is not in collection')
                indexing = reko.index_faces(
                    CollectionId = 'hackaton',
                    Image = {
                        'Bytes': image
                    },
                    MaxFaces = 1,
                    DetectionAttributes = ["DEFAULT"]
                )
                face_info = reko.detect_faces(
                    Image = {
                        'Bytes': image,
                    },
                    Attributes=["ALL"]
                )
                print(face_info["FaceDetails"][0]["AgeRange"], face_info["FaceDetails"][0]["Gender"])

                if indexing.get('FaceRecords'):
                    faceId=indexing.get('FaceRecords')[0]['Face']['FaceId']
                    #print(faceId)
                    #enviar a firebase
                    headers = {'Content-type': 'application/json'}
                    data = {"faceId":faceId, 'timestamp':time.time() }
                    print(time.time())
                    #print (data)
                    doc_ref_image_data.push({
                        "bucketname": "crowdlytics-1.appspot.com",
                        "cam-id": "1",
                        "filename" : file_list[oldest_i],
                        "AgeRange":(face_info["FaceDetails"][0]["AgeRange"]["High"] + face_info["FaceDetails"][0]["AgeRange"]["Low"])/2,
                        "Gender": face_info["FaceDetails"][0]["Gender"]["Value"],
                        "timestamp": time.time()
                    })
                    #r = requests.post("https://valo-hk.firebaseio.com/log.json", headers=headers, data=json.dumps(data))
            try:
                os.mkdir(faces_path + faceId)
            except:
                pass
            #created = str(int(creation[oldest_i]))
            #copyfile(image_path, faces_path + faceId + '/' + created + '.jpg')
            blob = bucket.blob(file_list[oldest_i])
            blob.upload_from_filename(image_path)
            os.remove(image_path)
        except Exception as e:
            if str(e) == 'An error occurred (InvalidParameterException) when calling the SearchFacesByImage operation: There are no faces in the image. Should be at least 1.':
                print('No faces in image, deleting')
                os.remove(image_path)
    else:
        print("No images, going to sleep for 1s")
        time.sleep(1)

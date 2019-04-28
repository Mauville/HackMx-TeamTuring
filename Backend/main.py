from flask import Flask, render_template
import io
import os, glob
import firebase_admin
from firebase_admin import storage, credentials, firestore, storage, db
import datetime
from flask import Flask
from flask import Markup
from flask import Flask
from flask import render_template


# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

app = Flask(__name__)
Vector = [0,0,0,0]
total = 0


@app.route('/')
def chart():
    counter = total
    labels = []
    values = [Vector[0],Vector[1],Vector[2],Vector[3], total ]
    return render_template('chart.html', values=values, labels=str(labels), counter=counter)

def image():
    global Vector
    cred = credentials.Certificate("crowdlytics-1-firebase-adminsdk-7bj5s-4af8c62b68.json")
    # Initialize the app with a service account, granting admin privileges
    app = firebase_admin.initialize_app(cred, {
    'storageBucket': 'crowdlytics-1.appspot.com',}, name='picname')
    client = vision.ImageAnnotatorClient()
    bucket = storage.bucket(app=app)

    firebase_admin.initialize_app(cred, {
        'storageBucket': 'crowdlytics-1.appspot.com',
        'databaseURL': 'https://crowdlytics-1.firebaseio.com/'
    })
    doc_ref_data = db.reference('/image-data')
    snapshot = doc_ref_data.get()
    dataIDS = []
    global total
    for element in snapshot:
        dataIDS.append(element)
        total +=1
    for element in dataIDS:
        blobname = snapshot[element]["filename"]

        blob = bucket.blob("picname")
        url = blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')
        image = types.Image()
        image.source.image_uri = url
        response = client.face_detection(image=image)
        faces = response.face_annotations

        personalV = processImage(faces)
        for i in range(4):
            Vector[i] += personalV[i]

    doc_ref_data = db.reference('/data')
    doc_ref_data.set({
        "face-id": "laskdjlaksdj",
        "cam-id": "1",
        "mood" : {"anger":Vector[0],"joy":Vector[1],"sorrow":Vector[2], "surprise":Vector[3]}
    })

    #return "okay"

def printHola():
    return("hola")

def processImage(faces):
    # Names of likelihood from google.cloud.vision.enums
    ResultVector = [0,0,0,0]
    for face in faces:
        #print('anger: {}'.format(likelihood_name[face.anger_likelihood]))
        if face.anger_likelihood > 2:
            ResultVector[0] += face.anger_likelihood/5
        #print('joy: {}'.format(likelihood_name[face.joy_likelihood]))
        if face.joy_likelihood > 2:
            ResultVector[1] += face.joy_likelihood/5
        #print('sorrow: {}'.format(likelihood_name[face.sorrow_likelihood]))
        if face.sorrow_likelihood > 2:
            ResultVector[2] += face.sorrow_likelihood/5
        #print('surprise: {}'.format(likelihood_name[face.surprise_likelihood]))
        if face.surprise_likelihood > 2:
            ResultVector[3] += face.surprise_likelihood/5
    return ResultVector

if __name__ == '__main__':
    app.run()

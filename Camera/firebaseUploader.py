import firebase_admin
from firebase_admin import credentials, firestore, storage, db

cred = credentials.Certificate('crowdlytics-1-firebase-adminsdk-7bj5s-4af8c62b68.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'crowdlytics-1.appspot.com',
    'databaseURL': 'https://crowdlytics-1.firebaseio.com/'
})
fs_client = firestore.client()
bucket = storage.bucket()
blob = bucket.blob('picname')
outfile='./findings/5fc64223-2b5c-430f-9c1c-1d8a88ef7175.jpg'
blob.upload_from_filename(outfile)
print("done")

doc_ref_image_data = db.reference('/image-data')
doc_ref_image_data.set({
    "bucketname": "crowdlytics-1.appspot.com",
    "cam-id": "entrance",
    "filename" : "picname"
})

doc_ref_data = db.reference('/data')
doc_ref_data.push({
    "face-id": "laskdjlaksdj",
    "cam-id": "1",
    "mood" : {"anger":1,"joy":0,"sorrow":0, "surprise":0}
})
doc_ref_analytics = db.reference('/analytics')
doc_ref_analytics.set({
    "active-visitors": 0,
    "mood-average": {"anger":0,"joy":1,"sorrow":0, "surprise":0},
    "mood" : {"anger":0,"joy":0,"sorrow":0, "surprise":0}
})

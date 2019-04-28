def face_finder():
    fps = 12
    #camera_addr="/dev/video0"
    camera_ip = "10.32.89.135"
    camera_addr="rtsp://admin:MAMG940901EN0@" + camera_ip
    #camera_addr='0'
    camera_addr="/dev/video1"

    print("Connecting to " + camera_ip + " ...")
    vcap = cv2.VideoCapture(camera_addr)
    vcap.set(20,1) #buffer size = 1
    vcap.set(cv2.CAP_PROP_FPS,60)  #camera.fps = fps Se tiene que configurar en la camara tambien
    vcap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    vcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    #vcap.set(cv2.CV_CAP_PROP_FPS, 60)
    vcap.set(15,-10)
    ret, frame = vcap.read() #hay que leerlo una vez para poder ver sus propiedades
    #print (frame.shape)
    frame_size_x = vcap.get(3) #guarda el tama;o del frame
    frame_size_y = vcap.get(4)
    print("Succesfully connected :D")


    actors = init_actors_multi()
    #print('initial actors shape' + str(actors['pos'].shape))
    boxes = init_boxes()
    face_detector = FaceDetector()
    counter = 0
    centers=[]
    while True:
        start = time.time()
        vcap.grab() #leer imagen de vcap
        ret, frame = vcap.retrieve(0)
        boxes = np.empty([0,4], dtype = int)
        if ret: #revisa que si recibio una imagen
            #print("vcap worked")
            #frame_small = cv2.resize(frame, (int(frame_size_x/factor), int(frame_size_y/factor)))
            frame_ann=frame
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bboxes = face_detector.predict(rgb_img, thresh=0.8)
            #frame_ann = annotate_image(frame, bboxes)
            #transform boxes into x,y,w,h starting in corner
            for bb in bboxes:
                x,y,w,h=bb[0:4]
                x = x - w/2
                y = y - w/2
                boxes = np.vstack([boxes,[int(x),int(y),int(w),int(h)]])
            #print('boxes: '+str(boxes))
        actors_new = scene_multi(boxes,actors,frame_size_x,frame_size_y)

        appeared = init_actors_multi()

        for i in range(actors_new['id'].shape[0]):
            unique = True
            box = actors_new['pos'][i,0]
            center_x,center_y = get_box_center(box)
            plot_traj(frame_ann,centers,center_x,center_y)
            x,y,w,h = expand_box(box, 2, frame_size_x, frame_size_y)
            cv2.rectangle(frame_ann,(x,y),(x+w,y+h),(0, 255, 0), 1)
            for j in range(actors['id'].shape[0]):
                #check if it existed already, else, append it to the appeared array
                if(actors_new['id'][i]==actors['id'][j]):
                    unique = False
            if unique:
                appeared['id'] = np.vstack([appeared['id'],actors_new['id'][i]])
                appeared['pos'] = np.vstack([appeared['pos'],[actors_new['pos'][i]]])

                print(x,y,w,h)
                face_crop = frame[y:y+h,x:x+w]
                cv2.imwrite('findings/' + str(time.time()) + '.jpg', face_crop)
                cv2.rectangle(frame_ann,(x,y),(x+w,y+h),(0, 255, 255), 4)
                counter+=1
                #print("Finding " + actors_new['id'][i][0] + ' saved')
        if len(centers)>0 and actors_new['id'].shape[0]==0:
            centers.pop(0)
        try:
            cv2.imshow("test",frame_ann)
        except:
            pass
        cv2.waitKey(1)

        print('appeared: '+ str(appeared['id']))
        actors = actors_new
        leisure_time = max(1/fps - (time.time() - start), 0)
        #print("leisure_time at " + str(fps) + "fps = " + str(leisure_time))
        print("counter",counter)
        time.sleep(leisure_time)

    return True

if __name__ == "__main__":
    import time
    import cv2
    import numpy as np
    import uuid
    from faced import FaceDetector
    from utils import *
    from faced.utils import annotate_image
    face_finder()

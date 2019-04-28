import numpy as np
import uuid
import cv2

def init_actors_multi():
    actors={
    'id':np.empty([0,1],dtype=str),
    'pos':np.random.rand(0,3,4)
    }
    return actors

def init_boxes():
    return np.empty([0,4])

def expand_box(box, expansion_factor, frame_size_x, frame_size_y):
    #Regresa la caja expandida por el factor sin salirse del framesize
    x,y,w,h = box[:]
    deltaw= (w * expansion_factor) - w
    deltah= (h * expansion_factor) - h
    x = int(max(x - deltaw/2, 0))
    y = int(max(y - deltah/2, 0))
    x2 = min(x + w + deltaw, frame_size_x)
    y2 = min(y + h + deltah, frame_size_y)
    w = int(x2 - x)
    h = int(y2 - y)
    box = np.zeros([4], dtype=int)
    box[:] = [x,y,w,h]
    return box
def get_box_center(box):
    x,y,w,h = box[:]
    x = x + w/2
    y = y + h/2
    return x,y
def plot_traj(frame_ann,centers,x,y):
    centers.append((x,y))
    if len(centers) > 30:
        centers.pop(0)
    #for c in centers:
        #circle(map,(x,y), diameter, c, -1)
        #cv2.circle(frame_ann, (int(c[0]), int(c[1])),2, (0, 0, 255),-1)
    return frame_ann

def scene (boxes, actors, frame_size_x,frame_size_y, min_closeness=0.3):
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
        #print('actors before filtering' + str(actors_uid))

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

def scene_multi(boxes, actors, frame_size_x,frame_size_y, min_closeness=0.15, min_size=16, max_size=2 ):
    def box_corners(box):
        #print(box)
        #print(box.shape)
        x,y,w,h = box
        x1_rel = x / frame_size_x
        y1_rel = y / frame_size_y
        x2_rel = (x + w) / frame_size_x
        y2_rel = (y + h) / frame_size_y
        box = [x1_rel, y1_rel, x2_rel, y2_rel]
        return box

    actors_uid = actors['id']
    actors = actors['pos']
    #print("actors shape input" + str(actors.shape))
    n_b = boxes.shape[0] #number of boxes
    n_a = actors.shape[0] #number of actors
    n_t = actors.shape[1] #number of times

    if n_a > 0:

        #convertir a esquinas normalizadas
        actors_corners = np.empty([n_a,3,4])
        for a in range(n_a):#para cada actor
            box = np.zeros([3,4])#cajitas para guardar las esquinas de los actores
            for t in range(n_t):#para cada tiempo
                #print(actors[a])
                #print(actors[a,t])
                actors_corners[a,t] = box_corners(actors[a,t,:])#convertir el actor [a, t]
        #print('actors_corners: ' + str(actors_corners))
        if boxes.shape[0]>0: #si hay predicciones
            boxes_corners = np.zeros([n_b,4])
            for b in range(n_b):#para cada caja
                boxes_corners[b] = box_corners(boxes[b])#convertir la caja
            #print('boxes_corners: '+ str(boxes_corners))

        #crear matriz de distancias de las cajas con todos los t de todos los actores
        distance = np.zeros([n_a,n_t,n_b])
        for b in range(n_b):
            for a in range(n_a):
                for t in range(n_t):
                    distance[a,t,b]=np.linalg.norm(np.subtract(boxes_corners[b],actors_corners[a,t]))
        #print('distance shape' + str(distance.shape))
        #print('distance: '+str(distance))

        #shift t in actors
        for a in range(n_a):
            actors[a,2] = actors[a,1]
            actors[a,1] = actors[a,0]
            actors[a,0] = np.zeros([1,4])

        #asignar boxes al actor mas cercano o crear nuevo actor
        #print('asignando boxes a actores')
        updated_actors = np.zeros([n_a,1],dtype=bool)#para saber si se actualizo
        for b in range(n_b):
            #print('distance: ' + str(distance[:,:,b]))
            d = np.min(distance[:,:,b]) #distance to the closest actor
            i = int(np.argmin(distance[:,:,b]) / n_t) #index of the closest actor
            #print('box ' + str(b) + ' closest : ' + str(d) )
            #print('box ' + str(b) + ' closest index: ' + str(i) )
            if d < min_closeness:
                #update actor
                #print('Updating actor ' + str(i))
                #print('actor[i,0]'+ str(actors))
                actors[i,0] = boxes[b]
                updated_actors[i] = 1
            else:
                #create new actor
                if boxes[b][2] > frame_size_x/min_size and boxes[b][3] > frame_size_x/min_size:
                    actors = np.append(actors,[[boxes[b],boxes[b],boxes[b]]], axis=0)
                    updated_actors=np.vstack([updated_actors,1])
                    #print('actors_uid: ' + str(actors_uid))

                    uniqueid = str(uuid.uuid4())
                    actors_uid = np.vstack([actors_uid,uniqueid])
        #borrar los que se volvieron puros ceros
        deletes = 0
        for a in range(n_a):
            #print('actors['+str(a)+']: ' +str(actors[a]))
            sum_actor = np.sum(actors[a-deletes])
            #print('sum_actor: '+ str(sum_actor))
            if sum_actor == 0:
                #print ('deleting this actor')
                actors = np.delete(actors, a-deletes, axis=0)
                actors_uid = np.delete(actors_uid, a-deletes, axis=0)
                deletes += 1
        #print('actors after deleting: '+ str(actors))
    else:
        #no habia actores, copiar boxes como actores nuevos
        for b in range(n_b):
            if boxes[b][2] > frame_size_x/min_size and boxes[b][3] > frame_size_x/min_size:#solo si son suficientemente grandes
                actors = np.append(actors,[[boxes[b],boxes[b],boxes[b]]], axis=0)
                uniqueid = str(uuid.uuid4())
                actors_uid = np.vstack([actors_uid,uniqueid])
    #print("actors shape output" + str(actors.shape) )
    actors={'id':actors_uid,'pos':actors}
    return actors

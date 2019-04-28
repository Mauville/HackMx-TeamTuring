import numpy as np
import uuid

actors={
    'id':np.empty([0,1],dtype=str),
    'pos':np.random.rand(0,3,4)
}
'''
actors['id'][0,:] = 'a'
actors['id'][1,:] = 'b'
actors['id'][2,:] = 'c'
'''


frame_size_x = 10
frame_size_y = 10

def scene_multi(boxes, actors, frame_size_x,frame_size_y, min_closeness=0.05):
    def box_corners(box):
        x,y,w,h = box
        x1_rel = x / frame_size_x
        y1_rel = y / frame_size_y
        x2_rel = (x + w) / frame_size_x
        y2_rel = (y + h) / frame_size_y
        box = [x1_rel, y1_rel, x2_rel, y2_rel]
        return box
    
    actors_uid = actors['id']
    actors = actors['pos']
    n_b = boxes.shape[0] #number of boxes
    n_a = actors.shape[0] #number of actors
    n_t = actors.shape[1] #number of times
    
    if n_a > 0:

        #convertir a esquinas normalizadas
        actors_corners = np.empty([n_a,3,4])
        for a in range(n_a):#para cada actor
            box = np.zeros([3,4])#cajitas para guardar las esquinas de los actores
            for t in range(n_t):#para cada tiempo
                actors_corners[a,t] = box_corners(actors[a,t])#convertir el actor [a, t]
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
            #print(distance[:,:,b])
            d = np.min(distance[:,:,b]) #distance to the closest actor
            i = np.argmin(distance[:,:,b]) % n_t #index of the closest actor
            #print('box ' + str(b) + ' closest : ' + str(d) )
            #print('box ' + str(b) + ' closest index: ' + str(i) )
            if d < min_closeness:
                #update actor
                #print('Updating actor ' + str(i))
                actors[i,0] = boxes[b]
                updated_actors[i] = 1
            else:
                #create new actor
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
            actors = np.append(actors,[[boxes[b],boxes[b],boxes[b]]], axis=0)
            uniqueid = str(uuid.uuid4())
            actors_uid = np.vstack([actors_uid,uniqueid])
        #print(actors)
    actors={'id':actors_uid,'pos':actors}
    return actors
for i in range(3):
    boxes = np.random.rand(2,4)
    actors = scene_multi(boxes,actors,frame_size_x, frame_size_y)
    print(actors['id'])
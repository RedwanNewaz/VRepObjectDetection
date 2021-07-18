import cv2
from pathlib import Path
from ObjectDetector import ObjectDetector


import sys
'''
To run this script first 
copy remoteApi.so file from your vrep installation directory: 
<vrep_installation_dir>/programming/remoteApiBindings/lib/lib/Linux/64Bit
to 
<vrep_installation_dir>/programming/remoteApiBindings/python/python
'''

sys.path.append('/opt/vrep/programming/remoteApiBindings/python/python')
sys.path.append('/opt/vrep/programming/remoteApiBindings/lib/lib/Linux/64Bit')



import vrep
import time
import cv2
import numpy as np

if __name__ == '__main__':

    labels = '../ObjectDetector/mscoco_label_map.pbtxt'
    detector = ObjectDetector('SSDmobileNet', labels)
    #demo image

    vrep.simxFinish(-1)

    clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)

    if clientID != -1:
        print('Connected to remote API server')

        # get vision sensor objects
        res, v0 = vrep.simxGetObjectHandle(clientID, 'v0', vrep.simx_opmode_oneshot_wait)
        res, v1 = vrep.simxGetObjectHandle(clientID, 'v1', vrep.simx_opmode_oneshot_wait)

        err, resolution, image = vrep.simxGetVisionSensorImage(clientID, v0, 0, vrep.simx_opmode_streaming)
        time.sleep(1)

        while (vrep.simxGetConnectionId(clientID) != -1):
            err, resolution, image = vrep.simxGetVisionSensorImage(clientID, v0, 0, vrep.simx_opmode_buffer)
            img = np.array(image, dtype=np.uint8)

            img = np.flipud(img)

            img = np.reshape(img, (resolution[0], resolution[1], 3))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            img = detector(img)
            cv2.imshow('frame', img)
            cv2.waitKey(1)
            if err == vrep.simx_return_ok:
                vrep.simxSetVisionSensorImage(clientID, v1, image, 0, vrep.simx_opmode_oneshot)
            elif err == vrep.simx_return_novalue_flag:
                print("no image yet")
            else:
                print("error")
    else:
        print("Failed to connect to remote API Server")
        vrep.simxFinish(clientID)
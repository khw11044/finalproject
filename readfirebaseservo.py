import os
import argparse
import numpy as np
import sys

#from motor import StepMotor
from servodrive import ServoMotor
import time
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
import importlib.util

cred = credentials.Certificate('drone-detection-js-firebase-adminsdk-4xh9r-3ba93b9ccd.json')
# Initialize the app with a service account, granting admin privileges
cred = credentials.Certificate('firestore-1add2-firebase-adminsdk-7vjg4-6c20413010.json')
firebase_admin.initialize_app(cred)
db = firestore.client()


panmotor=0
tiltmotor=0
currentstepstate=0
currentservostate=0
controlcount=0
controlservocount=0
while True:
    #t= StepMotor()
    t = ServoMotor()
    doc_ref_drone = db.collection(u'robot1').document(u'control')
    dic = doc_ref_drone.get().to_dict()
    try :
        pandatabase = int(dic['left_right'])
    except KeyError:
        print('no doc')
        pass
    
    try :
        controldatabase = int(dic['left_rightcontrol'])
        print(controldatabase)
    except KeyError:
        print('no doc')
        pass
    
    try :
        tiltdatabase = int(dic['up_down'])
        print(tiltdatabase)
    except KeyError:
        print('no doc')
        pass

    try :
        if pandatabase == 1:
            t.right()
        elif pandatabase == -1:
            t.left()
        elif pandatabase == 0:
            t.stop()
            controlcount = 0
            print('stop')
    except NameError :
        print('no left_right')
        pass
        
    try :
        if controldatabase == 1:
            controlcount =+1
        elif controldatabase == -1:
            controlcount =-1
            print(controlcount)

        else :
            controlcount = 0
    except NameError :
        print('no left_rightcontrol')
        pass
    
    
    try :
        if tiltdatabase == 1:
            controlservocount += 1
        elif tiltdatabase == -1:
            controlservocount -= 1
        else :
            controlservocount = 0
    except NameError :
        print('no up_down')
        pass
    
    if panmotor != controlcount :
        currentstepstate = controlcount - panmotor
        controlcount = panmotor
        print('currentstate : ',currentstepstate)
        
    if tiltmotor != controlservocount:
        currentservostate = controlservocount - tiltmotor
        controlservocount = tiltmotor
        print('currentservostate : ',currentservostate)
 
    if currentstepstate == 1 :
        t.right()
        currentstepstate = 0
    elif currentstepstate == -1 :
        t.left()
        currentstepstate = 0
    else :
        if currentstepstate > 1:
            for i in range(currentstepstate):
                t.right()
                currentstepstate = 0
        elif currentstepstate < -1 :
            for i in range(abs(currentstepstate)):
                t.left()
                currentstepstate = 0
                
    
    if currentservostate == 1 :
        t.up()
        print("up")
        currentservostate = 0
    elif currentservostate == -1 :
        t.down()
        print("down")
        currentservostate = 0
    else :
        if currentservostate > 1:
            for i in range(currentservostate):
                t.up()
                currentservostate = 0
        elif currentservostate < -1 :
            for i in range(abs(currentservostate)):
                t.down()
                currentservostate = 0
    

    
                

    
    
        



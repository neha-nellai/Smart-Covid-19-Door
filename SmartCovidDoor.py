# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
from smbus2 import SMBus
from mlx90614 import MLX90614
import RPi.GPIO as GPIO
from time import sleep
import tkinter as tk
from tkinter import *
import threading
import http.client as hp
import urllib
master=tk.Tk()
master.geometry("300x300")
my_text=Text(master,height=10)
my_text.config(state=NORMAL)
key = "YFBCCM43F0BNF97F"
count=0
#import _thread
#import threading

GPIO.setwarnings(False)
#initialize temperature sensor bus and gpio
bus = SMBus(1)
sensor = MLX90614(bus, address=0x5a)

#LED setup
'''greenLed = 8
redLed = 7
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(greenLed, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(redLed, GPIO.OUT, initial=GPIO.LOW)'''

#Servo motor setup
GPIO.setmode(GPIO.BOARD)
servoPin = 15
GPIO.setup(servoPin, GPIO.OUT)
pwm = GPIO.PWM(servoPin, 50)


GPIO.setmode(GPIO.BOARD)

Motor1A=36
Motor1B=38
Motor1E=40

GPIO.setup(Motor1A,GPIO.OUT)
GPIO.setup(Motor1B,GPIO.OUT)
GPIO.setup(Motor1E,GPIO.OUT)

GPIO.setmode(GPIO.BOARD)
trig=16
echo=18
GPIO.setup(trig,GPIO.OUT)
GPIO.setup(echo,GPIO.IN)

#IR sensor setup
ir = 10
GPIO.setmode(GPIO.BOARD)
GPIO.setup(ir, GPIO.IN)

def message(msg):
    my_text.insert("0.0",msg)
    my_text.pack()
    #print("hello")
    master.update_idletasks()
    master.update()
    
def message_temp(temperature):
    my_text.insert("0.0",str(temperature))
    my_text.pack()
    #print("hello")
    master.update_idletasks()
    master.update()
    
def delete():
    my_text.delete("0.0",END)
    msg=""
    master.after(0000,message(msg))
    #print("delete")

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	sleep(2)
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
		
		

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)


def openGate():
	print ("Turning motor on")
	GPIO.output(Motor1A,GPIO.LOW)
	GPIO.output(Motor1B,GPIO.HIGH)
	GPIO.output(Motor1E,GPIO.HIGH)
	
	sleep(7)
	GPIO.output(Motor1A,GPIO.LOW)
	GPIO.output(Motor1B,GPIO.LOW)
	GPIO.output(Motor1E,GPIO.LOW)
    
    
def closeGate():
	print ("Closing door")
	GPIO.output(Motor1A,GPIO.HIGH)
	GPIO.output(Motor1B,GPIO.LOW)
	GPIO.output(Motor1E,GPIO.HIGH)
	
	sleep(2)
	GPIO.output(Motor1A,GPIO.LOW)
	GPIO.output(Motor1B,GPIO.LOW)
	GPIO.output(Motor1E,GPIO.LOW)
    
def sanitize_hand():
    flag1=0
    GPIO.output(trig,True)
    time.sleep(0.00001)
    GPIO.output(trig,False)
    while GPIO.input(echo)==0:
        pulse_start=time.time()
    while GPIO.input(echo)==1:
        pulse_end=time.time()
    pulse_duration=pulse_end-pulse_start
    distance=pulse_duration*17150
    distance=round(distance+1.15,2)
    if(distance<12 and distance>5):
        print("Distance= ", distance, "cm. Dispensing sanitizer")
        pwm.start(2.5)
        pwm.ChangeDutyCycle(5)
        time.sleep(0.5)
        pwm.ChangeDutyCycle(7.5)
        time.sleep(0.5)
        pwm.ChangeDutyCycle(10)
        time.sleep(0.5)
        pwm.ChangeDutyCycle(12.5)
        time.sleep(0.5)
        pwm.ChangeDutyCycle(10)
        time.sleep(0.5)
        pwm.ChangeDutyCycle(7.5)
        time.sleep(0.5)
        pwm.ChangeDutyCycle(5)
        time.sleep(0.5)
        pwm.ChangeDutyCycle(2.5)
        time.sleep(0.5)
        pwm.stop()
        flag1=1
    else:
        print("Please keep your hand near the sanitizer")
        msg="Please keep your hand near the sanitizer"
        master.after(2000,message(msg))
        master.after(2000,delete())
        master.after(3000,lambda:msg.delete(0,END))
        sleep(1)
    return flag1
        
    
#Apply Algorithm
def applyLogic(label):
    #pwm.start(0)
    if (label=="No Mask"):
        msg="No Mask. Cannot open gate"
        master.after(2000,message(msg))
        master.after(2000,delete())
        #gateClose = threading.Thread(target=closeGate)
        #gateClose.start()
        #sendMessage("No Mask","Please wear mask!")
    else:
        msg="Mask detected"
        master.after(2000,message(msg))
        master.after(2000,delete())
        temp = getTempData()
        temp=int(sensor.get_obj_temp())
        params = urllib.parse.urlencode({'field1': temp, 'key':key })
        headers = {"Content-typZZe": "application/x-www-form-urlencoded","Accept": "text/plain"}
        conn = hp.HTTPConnection("api.thingspeak.com:80")
        try:
            conn.request("POST", "/update", params, headers)
            response = conn.getresponse()
            print(temp)
            print(response.status, response.reason)
            data = response.read()
        except:
            print("connection failed")
        task=threading.Thread(target=temp)
        var=tk.StringVar()
        
        master.update()
        #gateOpen = threading.Thread(target=openGate)
        #gateOpen.start()
        if temp >= 37:
            print(temp," High temp")
            msg=" is your temperature. High temperature, Gate closed."
            master.after(2000,message(msg))
            master.after(2000,message_temp(temp))
            master.after(2000,delete())
            master.after(3000,lambda:msg.delete(0,END))
            sleep(1)
        else:
            print(temp," is within acceptable range. Gate opening")
            #count+=1
            msg=" is your temperature. Temperature is within acceptable range. Gate opening"
            master.after(2000,message(msg))
            master.after(2000,message_temp(temp))
            master.after(2000,delete())
            openGate()
            flag=0
            while(flag==0):
                flag=sanitize_hand()
            while(GPIO.input(ir)):
                print(temp,"  Motion detected. Gate is still open")
            else:
                print(temp," Motion not detected. Gate was open for 2 secs")
                print("Gate is closing")
                msg="Gate closing"
                master.after(2000,message(msg))
                master.after(2000,delete())
                master.after(3000,lambda:msg.delete(0,END))
                closeGate()
        

def getTempData():
    temp = sensor.get_obj_temp()
    return temp

def closeEverything():
    closeGate()

def detect_mask(locs, preds, frame):
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
            
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        print(label)

        # include the probability in the label
        label_out = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        #temperature sensor data
        temp = getTempData()
        #temp = sensor.get_object_1()
        person_temp = "Temp: {:.1f}".format(temp)
        
        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label_out, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.putText(frame, person_temp, (endX-10, endY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
        return label     
        
        #_thread.start_new_thread(applyLogic, (label,))



# loop over the frames from the video stream
def run_video(detect_and_predict_mask):
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=1000)
        #cv2.normalize(frame, frame,0,255, cv2.NORM_MINMAX)
        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
            

        # loop over the detected face locations and their corresponding
        # locations
        label = detect_mask(locs, preds, frame)
        
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
        applyLogic(label)
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup

    cv2.destroyAllWindows()
    pwm.stop()
    GPIO.cleanup()
    vs.stop()
    


#main function
if __name__=="__main__":
    # load our serialized face detector model from disk
    prototxtPath = "/home/pi/PyMLX90614-0.0.4/Face-Mask_detection/face_detector/deploy.prototxt"
    weightsPath = "/home/pi/PyMLX90614-0.0.4/Face-Mask_detection/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    print("[INFO] Loading the model");
    # load the face mask detector model from disk
    maskNet = load_model("/home/pi/PyMLX90614-0.0.4/Face-Mask_detection/MaskDetector.h5")
    
    #opening gate
    #gate = threading.Thread(target=openGate)

    # initialize the video stream
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0, framerate=30).start()
    
    run_video(detect_and_predict_mask)
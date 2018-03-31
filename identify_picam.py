import tensorflow as tf, sys

# cam
from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import imutils
import time
import cv2
import random
import base64
import json
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
import os
import os.path
import re

import tarfile

import numpy as np
from six.moves import urllib


# button

import RPi.GPIO as GPIO
import time
import datetime




GPIO.setmode(GPIO.BCM)

GPIO.setup(18, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)


FLAGS = tf.app.flags.FLAGS

print("[INFO] cam sampling THREADED frames from `picamera` module...")
vs = PiVideoStream(resolution=(1024,768)).start()
time.sleep(2.0)
fps = FPS().start()

fn='/home/pi/Desktop/camlive.jpg'
#image_path = sys.argv[1]
#image_data=tf.gfile.FastGFile(image_path,'rb').read()
label_lines = [line.rstrip() for line 
					in tf.gfile.GFile("/home/pi/retrained_labels.txt")]


def cloud():
    #print("[INFO] cam sampling THREADED frames from `picamera` module...")
    frame = vs.read()
    frame = imutils.resize(frame,height=640,width=480)
    cv2.imwrite('/home/pi/Desktop/camlive2.jpg', frame)
    credentials = GoogleCredentials.get_application_default()
    service = discovery.build('vision', 'v1', credentials=credentials)

    with open('/home/pi/Desktop/camlive2.jpg', 'rb') as image:
        image_content = base64.b64encode(image.read())
        service_request = service.images().annotate(body={
            'requests': [{
                'image': {
                    'content': image_content.decode('UTF-8')
                },
                'features': [{
                    'type': 'LABEL_DETECTION',
                    'maxResults': 4
                }]
            }]
        })
        response = service_request.execute()
        response_obj = json.dumps(response)

        #print json.dumps(response_obj, indent=4, sort_keys=True)   #Print it out and make it somewhat pretty.
        temp_list = []

        for x in range(0,4):

            if(response['responses'][0]['labelAnnotations'][x]['score']>0.7):
                print (response['responses'][0]['labelAnnotations'][x]['description'])
                temp_list.append(response['responses'][0]['labelAnnotations'][x]['description'])
                #os.system("/usr/bin/pico2wave -w test.wav 'I believe this is a  "+response['responses'][0]['labelAnnotations'][x]['description']+"' | mplayer test.wav")
                #print (response['responses'][0]['labelAnnotations'][x]['score']) #Print it out and make it somewhat pretty.
            
       
        os.system("/usr/bin/pico2wave -w test.wav 'I see a' | mplayer test.wav")

        for y in range (0,len(temp_list)):
        	os.system("/usr/bin/pico2wave -w test.wav '"+temp_list[y]+"' | mplayer test.wav")





with tf.gfile.FastGFile("/home/pi/retrained_graph.pb",'rb') as f:
	graph_def=tf.GraphDef()
	graph_def.ParseFromString(f.read())
	_ = tf.import_graph_def(graph_def,name='')	



with tf.Session() as sess:
	#print("Session initialized.")
	os.system("/usr/bin/pico2wave -w test.wav 'Program loading, please wait' | mplayer test.wav")
	softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
	print("got tensor")
	os.system("/usr/bin/pico2wave -w test.wav 'I am ready now!' | mplayer test.wav")
	count = 0
	input_state = GPIO.input(18);
	input_state2 = GPIO.input(17);
	while True:


		if input_state == False:
			print('Button Pressed')
			frame = vs.read()
			frame = imutils.resize(frame, height=720,width=720)
			#print('Captured %dx%d image' % ( frame.shape[1], frame.shape[0]) )
			cv2.imwrite(fn, frame)
			image_data = tf.gfile.FastGFile('/home/pi/Desktop/camlive.jpg', 'rb').read()
			#print("read file")
			predictions= sess.run(softmax_tensor,\
			{'DecodeJpeg/contents:0':image_data})
			#predictions = np.squeeze(predictions)
			top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
			#top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
			
			for node_id in top_k:
			#node_id = top_k[0]
			#print(top_k)
				human_string = label_lines[node_id]
				score = predictions[0][node_id]
				
				if (score>0.55):
					print('%s (score= %.5f)' % (human_string,score))	
					os.system("/usr/bin/pico2wave -w test.wav 'I believe this is a  "+human_string+"' | mplayer test.wav")
					count = 0

			if(count<3):
				count= count+1
			elif(count==3):
				os.system("/usr/bin/pico2wave -w test.wav 'Please try again! ' | mplayer test.wav")
				count=0	

		if input_state2 == False:
			cloud()
			


				
			
			'''
			for node_id in top_k:

			#print(top_k)
				human_string = label_lines[node_id]
				score = predictions[0][node_id]
				print('%s (score= %.5f)' % (human_string,score))	
				#os.system("/usr/bin/pico2wave -w test.wav 'I believe this is a ... "+human_string+"' | mplayer test.wav")
			'''
			print('\n')

		

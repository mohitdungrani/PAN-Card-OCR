#!/bin/bash
from __future__ import print_function
from config import *
from utils.darknet_classify_image import *
from utils.tesseract_ocr import *
import utils.logger as logger
import sys
from PIL import Image
import time
import os
import cv2
import re
from operator import itemgetter
PYTHON_VERSION = sys.version_info[0]
OS_VERSION = os.name
import pandas as pd

class PanOCR():
	''' Finds and determines if given image contains required text and where it is. '''

	def init_vars(self):
		try:
			self.DARKNET = DARKNET
			
			self.TESSERACT = TESSERACT
			

			return 0
		except:
			return -1

	def init_classifier(self):
		''' Initializes the classifier '''
		try:
			if self.DARKNET:
			# Get a child process for speed considerations
				logger.good("Initializing Darknet")
				self.classifier = DarknetClassifier()
			
			if self.classifier == None or self.classifier == -1:
				return -1
			return 0
		except:
			return -1

	def init_ocr(self):
		''' Initializes the OCR engine '''
		try:
			if self.TESSERACT:
				logger.good("Initializing Tesseract")
				self.OCR = TesseractOCR()
			
			if self.OCR == None or self.OCR == -1:
				return -1
			return 0
		except:
			return -1

	def init_tabComplete(self):
		''' Initializes the tab completer '''
		try:
			if OS_VERSION == "posix":
				global tabCompleter
				global readline
				from utils.PythonCompleter import tabCompleter
				import readline
				comp = tabCompleter()
				# we want to treat '/' as part of a word, so override the delimiters
				readline.set_completer_delims(' \t\n;')
				readline.parse_and_bind("tab: complete")
				readline.set_completer(comp.pathCompleter)
				if not comp:
					return -1
			return 0
		except:
			return -1

	def prompt_input(self):
		
		
			filename = str(input(" Specify File >>> "))
		

	from utils.locate_asset import locate_asset

	def initialize(self):
		if self.init_vars() != 0:
			logger.fatal("Init vars")
		if self.init_tabComplete() != 0:
			logger.fatal("Init tabcomplete")
		if self.init_classifier() != 0:
			logger.fatal("Init Classifier")
		if self.init_ocr() != 0:
			logger.fatal("Init OCR")
	

	def find_and_classify(self, filename):
		''' find the required text field from given image and read it through tesseract.
		    Results are stored in a dicionary. '''
		start = time.time()
		

		#------------------------------Classify Image----------------------------------------#

                
		logger.good("Classifying Image")
		
		coords = self.classifier.classify_image(filename)
# 		print(coords + '********************')
		#lines=str(coords).split('\n')
		inf=[]
		for line in str(coords).split('\n'):
			if "sign" in line:
				continue
			if "photo" in line:
				continue
			if 'left_x' in line:
				info=line.split()
				left_x = int(info[3])
				top_y = int(info[5])
				inf.append((info[0],left_x,top_y))
		

		time1 = time.time()
		print("Classify Time: " + str(time1-start))

		# ----------------------------Crop Image-------------------------------------------#
		logger.good("Finding required text")
		cropped_images = self.locate_asset(filename, self.classifier, lines=coords)
		
		
		time2 = time.time()
		

		
		#----------------------------Perform OCR-------------------------------------------#
		
		ocr_results = None
		
		if cropped_images == []:
			logger.bad("No text found!")
			return None 	 
		else:
			logger.good("Performing OCR")
			ocr_results = self.OCR.ocr(cropped_images)
			#print(ocr_results)
			k=[]
			v=[]
			
			
			fil=filename+'-ocr'
			#with open(fil, 'w+') as f:
			for i in range(len(ocr_results)):
					
							v.append(ocr_results[i][1])
							k.append(inf[i][0][:-1])
							
			#k.insert(0,'Filename')
			#v.insert(0,filename)
			t=dict(zip(k, v))
			

		
		time3 = time.time()
		print("OCR Time: " + str(time3-time2))

		end = time.time()
		logger.good("Elapsed: " + str(end-start))
		print(t)
		return t
		
		
			
		#----------------------------------------------------------------#

	def __init__(self):
		''' Run PanOCR '''
		self.initialize()

def preProcessing(filename):
    image = cv2.imread(filename)
    h,w = image.shape[:2]
    diff = 0
    if h > w:
        if w > 800:
            diff = w-800
    else:
        if h > 700:
            diff = h-700
    image = cv2.resize(image, (w-diff,h-diff))
    original = image.copy()
    image = image[30:-30,30:-30]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (25,25), 0)
    blur = cv2.GaussianBlur(gray, (5,5), 1)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,3)
    thresh = cv2.threshold(th3, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Perform morph operations, first open to remove noise, then close to combine
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, noise_kernel, iterations=2)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, close_kernel, iterations=3)
    
    # Find enclosing boundingbox and crop ROI
    coords = cv2.findNonZero(close)
    x,y,w,h = cv2.boundingRect(coords)
    cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
    crop = original[y:y+h+60, x:x+w+60]
    if crop.shape[0] < 250 or crop.shape[1] < 300:
        crop2 = cv2.resize(crop, None,fx=2, fy=2,interpolation=cv2.INTER_CUBIC)
    elif crop.shape[0] > 500 or crop.shape[1] > 800:
            crop2 = cv2.resize(crop,None, fx=0.5,fy=0.5, interpolation= cv2.INTER_CUBIC)
    else:
        crop2 = crop.copy()
    cv2.imwrite(filename,crop2)

if __name__ == "__main__":
		extracter = PanOCR()
		tim = time.time()
		
		data=[]
		for filename in os.listdir("./input"):
			print(filename)
			filename='./input/'+filename
			preProcessing(filename)
			result=extracter.find_and_classify(filename)
			#print(df1)
			#df=df.append(df1)
			if result==None:
				continue
			else:
				data.append(result)
		
		df=pd.DataFrame(data)
		#print(df)
		df.to_csv (r'./results/ocr_result_ChopperID.csv', index = None, header=True,sep='\t')
		en = time.time()
		print('TOTAL TIME TAKEN',str(en-tim))

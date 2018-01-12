import cv2
import numpy as np

class ImageFeaturesExtraction:
	def __init__(self):
		self.surf=cv2.xfeatures2d.SURF_create(350,extended=True)
		self.image=None

	def image_read(self,path):
		self.image=cv2.imread(path)

	def image_crop(self):
		resized_image = cv2.resize(self.image, (450,450))
		#resized_crop_image=resized_image[12:448,78:370,:]
		return resized_image

	def applyCLAHE(self,img):
		# clr_lab=cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
		# l, a, b = cv2.split(clr_lab)
		# clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
		# cl=clahe.apply(l)
		# enhanced_image = cv2.merge((cl,a,b))
		# final = cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2BGR)
		# final_green=final[:,:,1]
		final_green=img[:,:,1]
		clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
		final_green=clahe.apply(final_green)
		return final_green

	def apply_dilation(self,img):
		strEl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
		dilateImg = cv2.dilate(img, strEl)
		return dilateImg

	def apply_threshold(self,img):
		retValue, threshImg = cv2.threshold(img, 220, 220, cv2.THRESH_BINARY)
		return threshImg

	def apply_median_filter(self,img):
		medianImg = cv2.medianBlur(img,5)
		return medianImg

	def getFeatures(self,green_channel):
		kp,des=self.surf.detectAndCompute(green_channel,None)
		return des

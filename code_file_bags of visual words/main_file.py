import numpy as np
import pandas as pd
from glob import glob
from image_features import *
from getDictionary import *
from matplotlib import pyplot as plt 
from trainSVM import *
from trainLG import *
from trainRandomForest import *
from sklearn.metrics import f1_score
import pickle as p

class Detect_Diabetic:
	def __init__(self):
		print('hello')
		self.feature_descriptor=[]
		self.image_counts=0
		self.img_des_extract=ImageFeaturesExtraction()
		self.traine_labels=np.array([],dtype=int)
		self.labels=pd.read_csv("/home/suraj/Documents/ML_Project/project/trainLabels.csv")

	def getImageDescriptor(self,path):
		for im_path in glob(path+"*"):
			im_name=im_path.split("/")[-1]
			im_name=im_name.split('.')[0]
			self.image_counts+=1
			#print(im_name)
			self.traine_labels=np.append(self.traine_labels,self.labels.loc[self.labels['image']==im_name]['level'].iloc[0])
			self.img_des_extract.image_read(im_path)
			crop_image=self.img_des_extract.image_crop()
			green_channel_img=self.img_des_extract.applyCLAHE(crop_image)
			# dilated_img=self.img_des_extract.apply_dilation(green_channel_img)
			# threshold=self.img_des_extract.apply_threshold(dilated_img)
			# median_filter_img=self.img_des_extract.apply_median_filter(threshold)
			self.feature_descriptor.append(self.img_des_extract.getFeatures(green_channel_img))
		return self.feature_descriptor,self.traine_labels

    


if __name__ == '__main__':
	correct_predicted=0
	correct_predicted_lg=0
	correct_predicted_rf=0
	dd=Detect_Diabetic()
	train_path="train_data/"
	features,labels=dd.getImageDescriptor(train_path)
	print(labels,features[0].shape)
	dict=CreateDictionary(n_clusters=20)
	dd_data=dict.createV_data(features)
	dict.create_cluster()
	dict.create_dictionary(dd.image_counts,features)
	#print(dict.dictionary)
	img_dict=dict.data_standardize()
	#print(img_dict)
	train_classifier=Train_Test_SVM()
	train_classifier.train_data(dict.dictionary,labels)
	train_classifier_lg=Train_Test_LG()
	train_classifier_lg.train_data(dict.dictionary,labels)
	train_classifier_rf=Train_Test_RF()
	train_classifier_rf.train_data(dict.dictionary,labels)
	test_path="test_data/"
	features1,labels1=Detect_Diabetic().getImageDescriptor(test_path)
	print(len(features1),labels1)
	predicted_labels=np.array([],dtype=int)
	predicted_labels_lg=np.array([],dtype=int)
	predicted_labels_rf=np.array([],dtype=int)
	for i in range(len(features1)):
		img_dict1=np.array([0 for i in range(dict.n_clusters)])
		dim=img_dict1.shape[0]
		test_ret=dict.kmeans_obj.predict(features1[i])
		for j in test_ret:
			img_dict1[j]+=1
		img_dict1=img_dict1.reshape(1,dim)
		img_dict1=dict.scale.transform(img_dict1)
		predicted_labels=np.append(predicted_labels,train_classifier.test_data(img_dict1))
		predicted_labels_lg=np.append(predicted_labels_lg,train_classifier_lg.test_data(img_dict1))
		predicted_labels_rf=np.append(predicted_labels_rf,train_classifier_rf.test_data(img_dict1))

	for i in range(0,len(predicted_labels)):
		if predicted_labels[i]==labels1[i]:
			correct_predicted+=1
		if predicted_labels_lg[i]==labels1[i]:
			correct_predicted_lg+=1
		if predicted_labels_rf[i]==labels1[i]:
			correct_predicted_rf+=1

	print("SVM result",correct_predicted,len(predicted_labels))
	print("LG result",correct_predicted_lg,len(predicted_labels_lg))
	print("RF result",correct_predicted_rf,len(predicted_labels_rf))

	print("SVM F1-Macro Score ",f1_score(labels1,predicted_labels,average='macro'))
	print("SVM F1-Micro Score ",f1_score(labels1,predicted_labels,average='micro'))
	print("SVM F1-weighted Score ",f1_score(labels1,predicted_labels,average='weighted'))
	print("LG F1-Macro Score ",f1_score(labels1,predicted_labels_lg,average='macro'))
	print("LG F1-Micro Score ",f1_score(labels1,predicted_labels_lg,average='micro'))
	print("LG F1-weighted Score ",f1_score(labels1,predicted_labels_lg,average='weighted'))
	print("RF F1-Macro Score ",f1_score(labels1,predicted_labels_rf,average='macro'))
	print("RF F1-Micro Score ",f1_score(labels1,predicted_labels_rf,average='micro'))
	print("RF F1-weighted Score ",f1_score(labels1,predicted_labels_rf,average='weighted'))

	p.dump(predicted_labels, open("svm_prediction.p", "wb"))
	p.dump(predicted_labels_rf, open("rf_prediction.p", "wb"))
	p.dump(predicted_labels_lg, open("lg_prediction.p", "wb"))
	p.dump(labels1, open("actual_prediction.p", "wb"))
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class CreateDictionary:
	def __init__(self,n_clusters):
		self.dictionary=None
		self.n_clusters=n_clusters
		self.kmeans_obj=KMeans(n_clusters=n_clusters)
		self.kmeans_label=None
		self.v_data=None

	def createV_data(self,des_data):
		self.v_data=np.vstack(des_data[0])
		print("hello")
		dd_len=len(des_data)
		for all_des in des_data[1:dd_len]:
			if all_des!=None:
				#print(all_des.shape)
				self.v_data=np.vstack((self.v_data,all_des))
		return self.v_data


	def create_cluster(self):
		self.kmeans_label=self.kmeans_obj.fit_predict(self.v_data)
		return self.kmeans_label

	def create_dictionary(self,no_images,desc_list):
		self.dictionary=np.array([np.zeros(self.n_clusters) for i in range(no_images)])
		prev_img=0
		for i in range(no_images):
			if(desc_list[i]!=None):
				l=len(desc_list[i])
				for j in range(l):
					label=self.kmeans_label[prev_img+j]
					self.dictionary[i][label]+=1
				prev_img+=l
		print("dictionary created")

	def data_standardize(self):
		self.scale=StandardScaler().fit(self.dictionary)
		self.dictionary = self.scale.transform(self.dictionary)
		return self.dictionary

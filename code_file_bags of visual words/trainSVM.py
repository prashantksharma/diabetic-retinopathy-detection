import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


class Train_Test_SVM:

	def __init__(self):
		self.svm_initializer=SVC()

	def train_data(self,data,labels):
		print("training svm")
		self.svm_initializer.fit(data,labels)
		print("Training Completed")

	def test_data(self,data):
		return self.svm_initializer.predict(data)
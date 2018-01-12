import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class Train_Test_RF:

	def __init__(self):
		self.rf_initializer=RandomForestClassifier(n_estimators=50, max_depth=20, random_state=1)

	def train_data(self,data,labels):
		print("training rf")
		self.rf_initializer.fit(data,labels)
		print("Training Completed")

	def test_data(self,data):
		return self.rf_initializer.predict(data)
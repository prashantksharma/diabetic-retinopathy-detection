import numpy as np
from sklearn.linear_model import LogisticRegression

class Train_Test_LG:

	def __init__(self):
		self.lg_initializer=LogisticRegression()

	def train_data(self,data,labels):
		print("training lg")
		self.lg_initializer.fit(data,labels)
		print("Training Completed")

	def test_data(self,data):
		return self.lg_initializer.predict(data)
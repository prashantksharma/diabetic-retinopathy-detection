

import numpy as np
import pandas as pd
from sklearn.svm import SVC
import csv
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

file="extracted_data.csv"

def get_input(data):

	for i in range(len(data)):
		data[i]=data[i]
	return data



def get_output(data):

  X=list()
  for i in range(len(data)):
    X.append(data[i].pop(-1))

  return X

   

def svc(input_data,output_data):

  model = SVC().fit(input_data, output_data)

  return model
    
def test(test_data,model):

  output=model.predict(test_data)

  return output

def data_conversion(file):

  with open(file, 'r') as f:
    reader=csv.reader(f)
    data=list(reader)

    for i in range(len(data)):
      for j in range(len(data[i])):
        data[i][j]=int(float(data[i][j]))
        #print (data[i][j])

  return data

   
def main():

  data=data_conversion(file)

  input_data=get_input(data)

  output_data=get_output(data)

  test_data=input_data[700:]

  test_data_op=output_data[700:]

  input_data=input_data[0:699]

  output_data=output_data[0:699]

  
  model=svc(input_data,output_data)


  output=test(test_data,model)

  accuracy = accuracy_score(test_data_op, output)

  print("svc_F1_score macro average",f1_score(test_data_op,output,average='macro'))
  print("svc_F1_score micro average",f1_score(test_data_op,output,average='micro'))
  print("svc_F1_score weighted average",f1_score(test_data_op,output,average='weighted'))
  print (accuracy)

  num_rows= output.shape

  index=[]

  for i in range(num_rows[0]):
  	index.append(i) 

  df = pd.DataFrame({"id":index,"predicted_class" : output})
  df.to_csv("output.csv", index=False)
    

    
  

if __name__ == "__main__":
    main()

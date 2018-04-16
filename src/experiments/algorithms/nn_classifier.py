'''
    Some imports, required for everything below.
'''
import csv
import joblib
import pickle
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from keras.optimizers import Adagrad
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout

print "All Libraries Loaded"

'''
  Changing directory of the script, so it can use the files in sister directories
'''
import os
import sys 
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
g_parentdir = os.path.dirname(parentdir)
sys.path.insert(0,g_parentdir) 




'''
    Read the data from the file and convert it into a numpy matrix
'''
#Read the CSV
f = open('output/experiments/EXP_001/out_exp003_wnut16_tweetNLP.csv')
data = csv.reader(f)

#Convert the data into a matrix
matrix_py = []
for row in data:
	matrix_py.append(row)

matrix_np = np.matrix(matrix_py)[1:]		#Removing 

print "Data encoded into a matrix. Size of matrix: ", matrix_np.shape



#Encode labels
matrix_np_labelled = np.zeros((matrix_np.shape[1],matrix_np.shape[0]))   #To store labelled data
encoders = []

'''
@diego: I'm feature_extraction encoders again since none of the existing ones have the encoding for most of the data. I looked at them all ;)
'''

for i in range(matrix_np.shape[1]):
    #Init a new encoder
    le = LabelEncoder()
    #Splice a column from the data
    column_data = np.array(matrix_np[:,i])
    #Fit and transform it.
    matrix_np_labelled[i] = le.fit_transform(column_data)

#Transpose the new matrix, so it resembles the original one
matrix_np_labelled = matrix_np_labelled.transpose()

print "Data Labels Encoded"



rows, cols = matrix_np_labelled.shape
traincount = int(0.8*rows)
train, test = matrix_np_labelled[:traincount], matrix_np_labelled[traincount:]
x_train = train[:,:-1]
y_train = train[:,-1]
x_test = test[:,:-1]
y_test = test[:,-1]

#One-Hot encoded output labels so they work well with a FeedForward Neural Network
def encode(x):
    return [1 if x == i else 0 for i in range(4)]
    
y_train = np.array([encode(row) for row in y_train])
y_test = np.array([encode(row) for row in y_test])

print "Data Split into Train and Test"

'''
    Declaring a simple NN as of now!
    @TODO: stamp out the desired NN
'''
model = Sequential()
# model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
# model.add(Dropout(0.2))
# model.add(LSTM(100, input_shape=(1,51)))
# model.add(Dropout(0.2))
model.add(Dense(100, input_shape=(51,)))
model.add(Dense(4))


opt = Adagrad()
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
print "\n"
print score
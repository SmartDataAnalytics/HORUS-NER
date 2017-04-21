# Create first network with Keras
import keras
import tensorflow as tf
from keras.backend import sparse_categorical_crossentropy
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
'''
1. Number of times pregnant 
2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test 
3. Diastolic blood pressure (mm Hg) 
4. Triceps skin fold thickness (mm) 
5. 2-Hour serum insulin (mu U/ml) 
6. Body mass index (weight in kg/(height in m)^2) 
7. Diabetes pedigree function 
8. Age (years) 
9. Class variable (0 or 1) 
'''
# load pima indians dataset
from horus import definitions
from horus.components.config import HorusConfig

config = HorusConfig()
samplefile = "/home/esteves/github/horus-models/output/features_dt_wnut16.csv"
sampletarget = "/home/esteves/github/horus-models/output/features_dt_wnut16_Y.csv"
#samplefile = "/home/esteves/github/horus-models/output/teste.txt"

dataset = numpy.loadtxt(samplefile, delimiter=",") #"pima-indians-diabetes.csv"
datasetT = numpy.loadtxt(sampletarget) #"pima-indians-diabetes.csv"
# split into input (X) and output (Y) variables
X = dataset[:,0:19]
Y = datasetT[:]

#X = dataset[:,0:8]
#Y = dataset[:,8]

#Yprime = keras.utils.np_utils.to_categorical(Y)

# create model
model = Sequential()
model.add(Dense(12, input_dim=19, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10,  verbose=2)
# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)
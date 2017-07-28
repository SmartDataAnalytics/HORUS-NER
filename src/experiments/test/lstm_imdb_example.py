# http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
# http://stackoverflow.com/questions/37307421/keras-blstm-for-sequence-labeling

# word embedding = map each movie review into a real vector domain
# words are encoded as real-valued vectors in a high dimensional space,
# where the similarity between words in terms of meaning translates
# to closeness in the vector space

# Keras provides a convenient way to convert positive integer representations of
# words into a word embedding by an Embedding layer.
# map each word onto a 32 length real valued vector.
import keras
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
from sklearn.metrics import f1_score

np.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(path='imdb.pkl', num_words=top_words)
print('x_train shape:', X_train.shape)
print('x_test shape:', X_test.shape)

#X_train = X_train[0:1000]
#X_test = X_test[0:1000]
# truncate and pad input sequences
max_review_length = 500
print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
print('x_train shape:', X_train.shape)
print('x_test shape:', X_test.shape)

from keras import metrics

print('Build model...')
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())
print('train...')

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
#model.fit(X_train, y_train, epochs=1, batch_size=128, validation_data=[X_test,y_test],
#       verbose=1, callbacks=[metrics])
#print metrics.f1s
#model.fit(X_train, y_train, nb_epoch=1, batch_size=128)
# Final evaluation of the model

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

exit(0)
################################################
# LSTM For Sequence Classification With Dropout
################################################

# Recurrent Neural networks like LSTM generally have the problem of overfitting.
# Dropout can be applied between layers using the Dropout Keras layer
'''
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
'''
# We can see dropout having the desired impact on training with a slightly slower trend in
# convergence and in this case a lower final accuracy.
# The model could probably use a few more epochs of training and may achieve a
# higher skill (try it an see).
# we can modify the first example to add dropout to the input and recurrent connections as follows:
'''
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
'''
# Dropout is a powerful technique for combating overfitting in your LSTM models and it                                                              is a good idea to try both
# methods, but you may bet better results with the gate-specific dropout provided in Keras.

################################################
# LSTM and Convolutional Neural Network For
# Sequence Classification
################################################
# Convolutional neural networks excel at learning the spatial structure in input data
# The IMDB review data does have a one-dimensional spatial structure in the sequence of words
# in reviews and the CNN may be able to pick out invariant features for good and bad sentiment.
'''
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
'''

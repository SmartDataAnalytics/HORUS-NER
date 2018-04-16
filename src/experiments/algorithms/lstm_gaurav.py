'''
    Some imports, required for everything below.
'''
import csv
import joblib
import pickle
import numpy as np
from keras import metrics

from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout
import keras.backend as K
from keras.preprocessing import sequence

print "All Libraries Loaded"

'''
    Read the data from the file and convert it into a numpy matrix
'''
#Read the CSV
f = open('/Users/esteves/Github/horus-models/output/experiments/EXP_001/wnut16.horus.conll')
data = csv.reader(f)

#Convert the data into a matrix
matrix_py = []
for row in data:
#     print row
#     raw_input()
    if row[7] == "1":
        continue
#         print roundw[3]
#         raw_input(row[3])
    matrix_py.append(row)

matrix_np = np.matrix(matrix_py)[1:]  #Removing the schema

print "Data encoded into a matrix. Size of matrix: ", matrix_np.shape

# Now, to encode the data into one hot labels, we splice the matrix per column
# (so as to get one features value across every data point, and then transform it and
# store it into another matrix.

matrix_np_labelled = np.zeros((matrix_np.shape[1], matrix_np.shape[0]))  # To store labelled data
encoders = []

# Encode labels
'''
@diego: I'm feature_extraction encoders again since none of the existing ones have the encoding for most of the data. I looked at them all ;)
'''

for i in range(matrix_np.shape[1]):
    # Init a new encoder
    le = LabelEncoder()
    # Splice a column from the data
    column_data = np.array(matrix_np[:, i])
    # Fit and transform it.
    matrix_np_labelled[i] = le.fit_transform(column_data)

# Transpose the new matrix, so it resembles the original one
matrix_np_labelled = matrix_np_labelled.transpose()


def encode_row(row):
    # returns a tuple with (encoding,label_encoding)
    #     print row
    encoding = []
    for x in row[:-1]:
        encoding.append(x)
    label = row[-1]
    #     print encoding
    #     if label == 1:
    #         print label
    #         raw_input()
    return [encoding, label]


row_dict = {}
for row in matrix_np_labelled:
    #     print row
    #     print int(row[1])
    try:
        row_dict[int(row[1])].append(encode_row(row))
    except:
        row_dict[int(row[1])] = []
        row_dict[int(row[1])].append(encode_row(row))
# raw_input()

print len(row_dict)
# raw_input()
print "hello"
sent = []
for key in row_dict:
    label = []
    encoding = []
    for t in row_dict[key]:
        label.append(t[1])
        encoding.append(t[0])
    sent.append([encoding, label])


# now converting each of the input sentence into a form of output label.

from keras.optimizers import Adagrad
from sklearn.preprocessing import OneHotEncoder

sent = np.asarray(sent)
rows, cols = sent.shape
# need to organize this as sentences before procedding as it needs split cases

traincount = int(0.8 * rows)
train, test = sent[:traincount], sent[traincount:]
x_train = [x[0] for x in train]
y_train = [x[1] for x in train]

x_test = [x[0] for x in test]
y_test = [x[1] for x in test]


# for x in x_train:
#     print x
#     print "**"
#     raw_input()
def encode(x):
    return [1 if x == i else 0 for i in range(4)]


# y_train = np.array([encode(row) for row in y_train])
# y_test = np.array([encode(row) for row in y_test])
maxlen = 0
for x in x_train:
    if len(x) > maxlen:
        maxlen = len(x)
print maxlen

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
y_train = sequence.pad_sequences(y_train, maxlen=maxlen)
y_test = sequence.pad_sequences(y_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)

'''
    Declaring a simple NN as of now!
'''

# model = Sequential()
# # model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
# # model.add(Dropout(0.2))
# # model.add(LSTM(100, input_shape=(1,51)))
# # model.add(Dropout(0.2))
# model.add(Dense(100, input_shape=(51,)))
# model.add(Dense(4))
print len(x_train[14])
max_features = 20000
print('Build model...')
print('x_train shape:', x_train.shape)

model = Sequential()
# model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2,input_shape=(37,51)))
model.add(Dense(37, activation='sigmoid'))

# # def f1(y_true, y_pred):
# #     print type(y_true)
# #     return K.mean(y_pred)
# #     y_true = y_true.eval()
# #     y_pred = y_pred.eval()
# #     f1 = f1_score(y_true, y_pred, average='weighted')
# #     return f1
# #     pass
# def recall(y_true, y_pred):
#     """Recall metric.
#     Only computes a batch-wise average of recall.
#     Computes the recall, a metric for multi-label classification of
#     how many relevant items are selected.
#     """
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall

# def fbeta_score(y_true, y_pred, beta=1):
#     """Computes the F score.
#     The F score is the weighted harmonic mean of precision and recall.
#     Here it is only computed as a batch-wise average, not globally.
#     This is useful for multi-label classification, where input samples can be
#     classified as sets of labels. By only using accuracy (precision) a model
#     would achieve a perfect score by simply assigning every class to every
#     input. In order to avoid this, a metric should penalize incorrect class
#     assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
#     computes this, as a weighted mean of the proportion of correct class
#     assignments vs. the proportion of incorrect class assignments.
#     With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
#     correct classes becomes more important, and with beta > 1 the metric is
#     instead weighted towards penalizing incorrect class assignments.
#     """
#     if beta < 0:
#         raise ValueError('The lowest choosable beta is zero (only precision).')

#     # If there are no true positives, fix the F score at 0 like sklearn.
#     if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
#         return 0

#     p = precision(y_true, y_pred)
#     r = recall(y_true, y_pred)
#     bb = beta ** 2
#     fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
#     return fbeta_score

# def precision(y_true, y_pred):
#     """Precision metric.
#     Only computes a batch-wise average of precision.
#     Computes the precision, a metric for multi-label classification of
#     how many selected items are relevant.
#     """
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision


# def fmeasure(y_true, y_pred):
#     """Computes the f-measure, the harmonic mean of precision and recall.
#     Here it is only computed as a batch-wise average, not globally.
#     """
#     return fbeta_score(y_true, y_pred, beta=1)

opt = Adagrad(lr=0.0001)
# model.compile(loss='categorical_crossentropy',
#               optimizer=opt,
#               metrics=['accuracy'])
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())
print('train gaurav...')

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)

predictions = model.predict(x_test, batch_size=256)
score = model.evaluate(x_test, y_test, batch_size=256)
print metrics.fmeasure(y_test, predictions)
print metrics.categorical_accuracy(y_test, predictions)
print score
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from random import random
from keras.callbacks import Callback
from keras.engine import InputLayer, Input
from keras.preprocessing.sequence import pad_sequences
from numpy import array
from numpy import cumsum
from matplotlib import pyplot
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dropout, Merge, Activation, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Flatten
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
import cPickle as pickle
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.config import HorusConfig
from src.core.util import definitions
from src.core.util.definitions import PLONone_label2index

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

config = HorusConfig()


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print("f1:%f prec:%f recall %f" % (_val_f1, _val_precision, _val_recall))
        return


metrics = Metrics()

def load_dumps_in_memory(_file):
    f = open(_file, 'rb')
    dump = pickle.load(f)
    f.close()
    return dump

def get_sequence(n_timesteps):
	# create a sequence of random numbers in [0,1]
	X = array([random() for _ in range(n_timesteps)])
	# calculate cut-off value to change class values
	limit = n_timesteps/4.0
	# determine the class outcome for each item in cumulative sequence
	y = array([0 if x < limit else 1 for x in cumsum(X)])
	# reshape input and output data to be suitable for LSTMs
	X = X.reshape(1, n_timesteps, 1)
	y = y.reshape(1, n_timesteps, 1)
	return X, y

def get_lstm_model(n_timesteps, backwards):
	model = Sequential()
	model.add(LSTM(20, input_shape=(n_timesteps, 1), return_sequences=True, go_backwards=backwards))
	model.add(TimeDistributed(Dense(1, activation='sigmoid')))
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

def get_bi_lstm_model(n_timesteps, mode):
	model = Sequential()
	model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_timesteps, 1), merge_mode=mode))
	model.add(TimeDistributed(Dense(1, activation='sigmoid')))
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

def train_model(model, n_timesteps):
	loss = list()
	for _ in range(250):
		# generate new random sequence
		X,y = get_sequence(n_timesteps)
		# fit model for one epoch on this sequence
		hist = model.fit(X, y, epochs=1, batch_size=1, verbose=0)
		loss.append(hist.history['loss'][0])
	return loss

def encode(x, n):
    result = np.zeros(n)
    result[x] = 1
    return result

def score2(yh, pr):
    #real-no-encoding x predicted
    coords = [np.where(yhh > 0)[0][0] for yhh in yh]
    yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    ypr = [prr[co:] for prr, co in zip(pr, coords)]
    fyh = [c for row in yh for c in row]
    fpr = [c for row in ypr for c in row]
    print('sets...')
    print set(fyh)
    print set(fpr)
    return fyh, fpr

def get_horus_embeedings_layer(MAX_SEQUENCE_LENGTH):
    config.logger.info('get HORUS embeddings')
    enc_word = joblib.load(config.dir_encoders + definitions.encoder_int_words_name)
    file = '/home/esteves/github/horus-ner/data/output/EXP_005/_ner.txt.horus_config_29.pkl'
    dump_configs = load_dumps_in_memory(file)
    X1_sentence = dump_configs[2]
    word_index = list(enc_word.classes_)
    print(len(word_index))
    EMB=definitions.FEATURES_HORUS_BASIC_AND_CNN_BEST_STANDARD
    EMBEDDING_DIM = len(EMB)
    embeddings_index = {} #np.zeros((60000, EMBEDDING_DIM))
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for X in X1_sentence:
        for index, row in X.iterrows():
            word_id=row[151]
            word=enc_word.inverse_transform(word_id)
            f = []
            for s in row.loc[EMB].tolist():
                if type(s) is str:
                    f.extend([float(s.replace('O', '0'))])
                else:
                    f.extend([s])
            embedding_matrix[word_id]=f
            #embeddings_index[word] = f  # brown_clusters = 4


    #embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    #for word, i in word_index.items():
    #    embedding_vector = embeddings_index.get(word)
    #    if embedding_vector is not None:
    #        # words not found in embedding index will be all-zeros.
    #        embedding_matrix[i] = embedding_vector
    #    else:
    #        print('word not found: ' + word)

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                mask_zero = True,
                                trainable = True)

    config.logger.info('embedding created')
    return embedding_layer

raw = open('/home/esteves/github/horus-ner/data/datasets/Ritter/ner.txt', 'r').readlines()

all_x = []
point = []
for line in raw:
    stripped_line = line.strip().split('\t')
    if len(stripped_line) == 2:
        if stripped_line[1]=='O':
            y='O'
        elif stripped_line[1] in definitions.NER_RITTER_LOC:
            y='LOC'
        elif stripped_line[1] in definitions.NER_RITTER_ORG:
            y='ORG'
        elif stripped_line[1] in definitions.NER_RITTER_PER:
            y='PER'
        else:
            y='O'
        point.append([stripped_line[0], y])
    else:
        point.append(stripped_line)
    if line == '\n':
        all_x.append(point[:-1])
        point = []
all_x = all_x[:-1]

lengths = [len(x) for x in all_x]
print 'Input sequence length range: ', max(lengths), min(lengths)

#short_x = [x for x in all_x if len(x) < 64]
short_x = all_x

X = [[c[0] for c in x] for x in short_x]
y = [[c[1] for c in y] for y in short_x]

all_text = [c for x in X for c in x]

words = list(set(all_text))
word2ind = {word: index for index, word in enumerate(words)}
ind2word = {index: word for index, word in enumerate(words)}
labels = list(set([c for x in y for c in x]))
label2ind = {label: (index + 1) for index, label in enumerate(labels)}
ind2label = {(index + 1): label for index, label in enumerate(labels)}
print 'Vocabulary size:', len(word2ind), len(label2ind)

label2ind = definitions.PLONone_label2index
ind2label = definitions.PLONone_index2label

maxlen = max([len(x) for x in X])
max_features = len(word2ind)
embedding_size = 128
hidden_size = 32
batch_size = 32
epochs=50
verbose=2
out_size = len(label2ind) + 1
print 'Maximum sequence length:', maxlen
max_label = max(label2ind.values()) + 1
HORUS = True


if HORUS is False:
    X_enc = [[word2ind[c] for c in x] for x in X]
    y_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y]
    y_enc = [[encode(c, max_label) for c in ey] for ey in y_enc]

else:
    enc_word = joblib.load(config.dir_encoders + definitions.encoder_int_words_name)
    word_index = list(enc_word.classes_)
    print 'Vocabulary size:', len(word_index), len(label2ind)

    file = '/home/esteves/github/horus-ner/data/output/EXP_005/_ner.txt.horus_config_2.pkl'
    dump_configs = load_dumps_in_memory(file)
    X1_sentence = dump_configs[2]
    Y1_sentence = dump_configs[3]

    maxlen = max([len(x) for x in X1_sentence])

    X_enc2 =[df[151] for df in X1_sentence]
    y_enc2 = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in Y1_sentence]
    y_enc2 = [[encode(c, max_label) for c in ey] for ey in y_enc2]

    word_ids_file=[]
    [word_ids_file.extend(ws) for ws in X_enc2]
    word_ids_file=set(word_ids_file)
    print 'Vocabulary size:', len(word_ids_file), len(label2ind)

    max_features2 = len(word_index)

    #X_enc = X_enc2 # tem um problema com is ids dos tokens, comente esse e descomente abaixo
    X_enc = [[word2ind[c] for c in x] for x in X]
    y_enc = y_enc2

'''
from this part, same process ---------------------------------------------------
'''

X_enc = pad_sequences(X_enc, maxlen=maxlen)
y_enc = pad_sequences(y_enc, maxlen=maxlen)

X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, test_size=0.2, random_state=42)
print 'Training and testing tensor shapes:', X_train.shape, X_test.shape, y_train.shape, y_test.shape

if HORUS:
    emb = get_horus_embeedings_layer(maxlen)
else:
    emb = Embedding(max_features, embedding_size, input_length=maxlen, mask_zero=True)

config.logger.info('setting up NN')
model = Sequential()
model.add(emb)
model.add(Bidirectional(LSTM(hidden_size, return_sequences=True), merge_mode='concat'))
model.add(TimeDistributed(Dense(out_size)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
print(model.summary())
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Raw test score:', score)
pr = model.predict_classes(X_train)
yh = y_train.argmax(2)
fyh, fpr = score2(yh, pr)
#print 'Training accuracy:', accuracy_score(fyh, fpr)
#print 'Training confusion matrix:'
#print confusion_matrix(fyh, fpr)
#print(precision_recall_fscore_support(fyh, fpr))
pr = model.predict_classes(X_test)
yh = y_test.argmax(2)
fyh, fpr = score2(yh, pr)
#print 'Testing accuracy:', accuracy_score(fyh, fpr)
#print 'Testing confusion matrix:'
#print confusion_matrix(fyh, fpr)
P, R, F, S = precision_recall_fscore_support(fyh, fpr)
name='test_NN.txt'
f_key=1
ds1_name='ritter'
ds2_name=ds1_name
experiment_folder='EXP_007/'
header = 'cross-validation\tconfig\trun\tlabel\tprecision\trecall\tf1\tsupport\talgo\tdataset1\tdataset2\ttask\n'
line = '%s\t%s\t%s\t%s\t%.5f\t%.5f\t%.5f\t%s\t%s\t%s\t%s\t%s\n'
out_file = open(config.dir_output + experiment_folder + name, 'w+')
out_file.write(header)
for k in range(0,3):
    out_file.write(line % ('False', str(f_key), '1', definitions.PLO_index2label.get(k+1),
                           P[k], R[k], F[k], str(S[k]), 'LSTM', ds1_name, ds2_name, 'NER'))



config.logger.info('DONE with embeedings!')
'''
CONCATENANDO AS LAYERS
'''
model1 = Sequential()
model1.add(Embedding(input_dim=max_features, output_dim=embedding_size, input_length=maxlen, mask_zero=True))
model2 = Sequential()
model2.add(InputLayer(input_shape=(maxlen, X_train.shape[2] - 1)))
model = Sequential()
model.add(Merge([model1, model2], mode='concat'))
model.add(Dense(1))
model.add(LSTM(hidden_size, return_sequences=True, input_shape=(maxlen, X_train.shape[2] - 1)))
model.add(TimeDistributed(Dense(out_size)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit([X_train[:, :, 0], X_train[:, :, 1:X_train.shape[2]]], y_train, epochs=epochs, verbose=verbose,
          batch_size=batch_size,
          validation_data=([X_test[:, :, 0], X_test[:, :, 1:X_train.shape[2]]], y_test))
score = model.evaluate([X_test[:, :, 0], X_test[:, :, 1:X_train.shape[2]]], y_test,
                       batch_size=batch_size, verbose=verbose)
print('Raw test score:', score)
pr = model.predict_classes(X_train)
yh = y_train.argmax(2)
fyh, fpr = score2(yh, pr)
pr = model.predict_classes(X_test)
yh = y_test.argmax(2)
fyh, fpr = score2(yh, pr)
P, R, F, S = precision_recall_fscore_support(fyh, fpr)
name='test_NN_2layers_concat.txt'
f_key=1
ds1_name='ritter'
ds2_name=ds1_name
experiment_folder='EXP_007/'
header = 'cross-validation\tconfig\trun\tlabel\tprecision\trecall\tf1\tsupport\talgo\tdataset1\tdataset2\ttask\n'
line = '%s\t%s\t%s\t%s\t%.5f\t%.5f\t%.5f\t%s\t%s\t%s\t%s\t%s\n'
out_file = open(config.dir_output + experiment_folder + name, 'w+')
out_file.write(header)
for k in range(0,3):
    out_file.write(line % ('False', str(f_key), '1', definitions.PLO_index2label.get(k+1),
                           P[k], R[k], F[k], str(S[k]), 'LSTM', ds1_name, ds2_name, 'NER'))

























exit(0)

#####################################################################
# HORUS
#############################################################
EMBEDDING_DIM=len(definitions.FEATURES_HORUS)
config = HorusConfig()
#horus_embeedings
file = '/home/esteves/github/horus-ner/data/output/EXP_005/_ner.txt.horus_config_29.pkl'
dump_configs = load_dumps_in_memory(file)
X1_sentence = dump_configs[2]
Y1_sentence = dump_configs[3]
embedding_matrix = np.zeros((60000, EMBEDDING_DIM))

X1_sentence_just_w = []
for X in X1_sentence:
    sent=[]
    for index, row in X.iterrows():
        word=row[151]
        sent.append(word)
        f=[]
        for s in row.loc[definitions.FEATURES_HORUS].tolist():
            if type(s) is str:
                f.extend([float(s.replace('O','0'))])
            else:
                f.extend([s])

        embedding_matrix[word] = f #brown_clusters = 4
    X1_sentence_just_w.append(sent)
vocab_size = len(embedding_matrix)
print(vocab_size)
print(embedding_matrix[0])
print(np.array(X1_sentence_just_w).shape)
#print(X1_sentence.shape)
#print(np.array(X1_sentence).shape)

if 1==2:
    file = '/home/esteves/github/horus-ner/data/output/EXP_005/_ner.txt.horus_config_2.pkl'
    dump_configs = load_dumps_in_memory(file)
    X1_sentence = dump_configs[2]
    Y1_sentence = dump_configs[3]

#X1_sentence = X1_sentence_just_w

lengths = [len(x) for x in X1_sentence]
maxsent=max(lengths)
out_size = len(PLONone_label2index) + 1
max_label = max(PLONone_label2index.values()) + 1

#[s.replace('O', 0, inplace=True) for s in X1_sentence]
#[[float(k) for k in s] for s in X1_sentence]
y_ids = [[PLONone_label2index[c] for c in ey] for ey in Y1_sentence]
y_enc = [[0] * (maxsent - len(ey)) + [PLONone_label2index[c] for c in ey] for ey in Y1_sentence]
y_enc = [[encode(c, max_label) for c in ey] for ey in y_enc]

X_lstm = pad_sequences(X1_sentence, maxlen=maxsent)
y_lstm = pad_sequences(y_enc, maxlen=maxsent)
#y_lstm = pad_sequences(y_ids, maxlen=maxlen)

hidden_size=32
embedding_size=128
batch_size=128
epochs=200
verbose=2
# size of dictionary
mf=100
n_timesteps = 43
results = DataFrame()

model = Sequential()

embedding_layer = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM,
                            input_length=maxsent, trainable=False) #weights=[embedding_matrix]
#model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(maxlen, 15), merge_mode='concat'))
#model.add(embedding_layer)
#model.add(Flatten())
model.add(LSTM(hidden_size, return_sequences=True, input_shape=(maxsent,X_lstm.shape[2]))) #maxlen, X_lstm.shape[2]
#model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(out_size)))
model.add(Activation('softmax'))

if 1==1:
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_lstm, y_lstm, epochs=epochs, batch_size=batch_size, verbose=verbose,
              validation_data=(X_lstm, y_lstm)
              ) #,callbacks=[metrics]
    score=model.evaluate(X_lstm, y_lstm, batch_size=batch_size, verbose=verbose)
    print('Raw test score:', score)
    pr = model.predict_classes(X_lstm, verbose=2)
    yh = y_lstm.argmax(2)  # no encoding
    fyh, fpr = score2(yh, pr)
    print(' - confusion matrix:')
    print(confusion_matrix(fyh, fpr))
    print(' - precision, recall, f1, support:')
    P, R, F, S = precision_recall_fscore_support(fyh, fpr)
    f_key=1
    ds1_name=''
    ds2_name=''
    name='test_NN.txt'
    experiment_folder='EXP_007/'
    header = 'cross-validation\tconfig\trun\tlabel\tprecision\trecall\tf1\tsupport\talgo\tdataset1\tdataset2\ttask\n'
    line = '%s\t%s\t%s\t%s\t%.5f\t%.5f\t%.5f\t%s\t%s\t%s\t%s\t%s\n'
    out_file = open(config.dir_output + experiment_folder + name, 'w+')
    out_file.write(header)
    for k in range(len(P)-1):
        out_file.write(line % ('False', str(f_key), '1', definitions.PLO_index2label.get(k + 1),
                               P[k], R[k], F[k], str(S[k]), 'DT', ds1_name, ds2_name, 'NER'))

#print (metrics.val_f1s)
exit(0)
print('-------------------------------------------')
# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(maxlen,), dtype='int32')
kernel_size=2
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, kernel_size, activation='relu')(embedded_sequences)
x = MaxPooling1D(kernel_size)(x)
x = Conv1D(128, kernel_size, activation='relu')(x)
x = MaxPooling1D(kernel_size)(x)
x = Conv1D(128, kernel_size, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(PLONone_label2index) + 1, activation='softmax')(x)
from keras.models import Model
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(X_lstm, y_lstm,
          batch_size=128,
          epochs=5,
          validation_data=(X_lstm, y_lstm))
exit(0)






results = DataFrame()
# lstm forwards
model = get_lstm_model(n_timesteps, False)
results['lstm_forw'] = train_model(model, n_timesteps)
# lstm backwards
model = get_lstm_model(n_timesteps, True)
results['lstm_back'] = train_model(model, n_timesteps)
# bidirectional concat

# line plot of results
results.plot()
pyplot.show()
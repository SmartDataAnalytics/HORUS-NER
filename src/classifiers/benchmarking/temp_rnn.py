
import os

from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib

from src.config import HorusConfig
from src.core.util import definitions

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import matplotlib.pyplot as plt


plt.style.use('ggplot')

from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix, precision_recall_fscore_support

import numpy as np
from keras.engine import InputLayer
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Merge

import cPickle as pickle


embedding_size = 128
hidden_size = 32
batch_size = 128
epochs = 50
verbose = 0

KLASSES = {1: "LOC", 2: "ORG", 3: "PER", 4: "O"}
KLASSES2 = {"LOC": 1, "ORG": 2, "PER": 3, "O": 4}

config = HorusConfig()
enc_word = joblib.load(config.dir_encoders + definitions.encoder_int_words_name)


def encode(x, n):
    result = np.zeros(n)
    result[x] = 1
    return result

def convert_lstm_shape(ds, y):
    Xclean = [[[c[3], c[4], c[10], c[12], c[13], c[17], c[18], c[20], c[21]] for c in x] for x in ds]

    all_text = [c[0] for x in ds for c in x]
    all_text.extend([c[1] for x in ds for c in x])

    words = list(set(all_text))  # distinct tokens
    word2ind = {word: index for index, word in enumerate(words)}  # indexes of words
    ind2word = {index: word for index, word in enumerate(words)}
    #labels = list(set([c for x in y for c in x]))
    label2ind = KLASSES2
    ind2label = KLASSES
    print('Vocabulary size:', len(word2ind), len(label2ind))
    lengths = [len(x) for x in ds]
    maxlen = max(lengths)
    print('min sentence / max sentence: ', min(lengths), maxlen)

    X_enc = [[[word2ind[c[0]], word2ind[c[1]], c[2], c[3], c[4], c[5], c[6], c[7], c[8]] for c in x] for x in ds]

    max_label = max(label2ind.values()) + 1
    y_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y]
    y_enc = [[encode(c, max_label) for c in ey] for ey in y_enc]

    max_features = len(word2ind)
    out_size = len(label2ind) + 1

    return ds, y_enc, max_features, out_size, maxlen

def load_dumps_in_memory(_file):
    f = open(_file, 'rb')
    dump = pickle.load(f)
    f.close()
    return dump

def score2(yh, pr):
    #real-no-encoding x predicted
    coords = [np.where(yhh > 0)[0][0] for yhh in yh]
    yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    ypr = [prr[co:] for prr, co in zip(pr, coords)]
    fyh = [c for row in yh for c in row]
    fpr = [c for row in ypr for c in row]
    print set(fyh)
    print set(fpr)
    return fyh, fpr

def run_lstm(Xtr, Xte, ytr, yte, max_features, max_features2, out_size, embedding_size, hidden_size, batch_size, epochs=50, verbose = 0, maxsent = 0):

    print('Training and testing tensor shapes:', Xtr.shape, Xte.shape, ytr.shape, yte.shape)

    mf = max(max_features, max_features2)

    model1 = Sequential()
    model1.add(Embedding(input_dim=mf, output_dim=embedding_size, input_length=maxsent, mask_zero=True))

    model2 = Sequential()
    model2.add(InputLayer(input_shape=(maxsent, Xtr.shape[2] - 1)))

    model = Sequential()
    model.add(Merge([model1, model2], mode='concat'))
    model.add(Dense(1))

    model.add(Bidirectional(LSTM(hidden_size, return_sequences=True, input_shape=(maxsent, Xtr.shape[2] - 1))))
    model.add(TimeDistributed(Dense(out_size)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    print('train...')

    model.fit([Xtr[:, :, 0], Xtr[:, :, 1:Xtr.shape[2]]], ytr, epochs=epochs, verbose=verbose, batch_size=batch_size,
              validation_data=([Xte[:, :, 0], Xte[:, :, 1:Xtr.shape[2]]], yte))
    score = model.evaluate([Xte[:, :, 0], Xte[:, :, 1:Xtr.shape[2]]], yte, batch_size=batch_size, verbose=verbose)

    print('Raw test score:', score)
    pr = model.predict_classes([Xtr[:, :, 0], Xtr[:, :, 1:Xtr.shape[2]]], verbose=verbose)
    yh = ytr.argmax(2)  # no encoding
    fyh, fpr = score2(yh, pr)
    print('Training...')
    print(' - accuracy:', accuracy_score(fyh, fpr))
    print(' - confusion matrix:')
    print(confusion_matrix(fyh, fpr))
    print(' - precision, recall, f1, support:')
    print(precision_recall_fscore_support(fyh, fpr))

    pr = model.predict_classes([Xte[:, :, 0], Xte[:, :, 1:Xte.shape[2]]], verbose=verbose)
    yh = yte.argmax(2)
    fyh, fpr = score2(yh, pr)
    print('Testing...')
    print(' - accuracy:', accuracy_score(fyh, fpr))
    print(' - confusion matrix:')
    print(confusion_matrix(fyh, fpr))
    print(' - precision, recall, f1, support:')
    print(precision_recall_fscore_support(fyh, fpr))
    print('----------------------------------------------------------------------------------')

file = '/home/esteves/github/horus-ner/data/output/EXP_005/_ner.txt.horus_config_29.pkl'
dump_configs = load_dumps_in_memory(file)

ds1_config_name = dump_configs[0]
ds1_key = dump_configs[1]
X1_sentence = dump_configs[2]
Y1_sentence = dump_configs[3]
X1_token = dump_configs[4]
Y1_token = dump_configs[5]
X1_crf = dump_configs[6]
_Y1_sentence = dump_configs[7]

#X1_lstm, y1_lstm, max_features_1, out_size_1, maxlen_1 = convert_lstm_shape(X1_sentence, Y1_sentence)

lengths = [len(x) for x in X1_sentence]
maxlen = max(lengths)
minlen = min(lengths)
max_label = max(KLASSES2.values()) + 1
y_enc = [[0] * (maxlen - len(ey)) + [KLASSES2[c] for c in ey] for ey in Y1_sentence]
y_enc = [[encode(c, max_label) for c in ey] for ey in y_enc]
max_features = len(enc_word.classes_)
out_size = len(KLASSES2) + 1

max_of_sentences = max(maxlen, maxlen)
X_lstm = pad_sequences(X1_sentence, maxlen=maxlen)
y_lstm = pad_sequences(y_enc, maxlen=maxlen)
Xtr, Xte, ytr, yte = train_test_split(X_lstm, y_lstm, test_size=0.3, random_state=42)  # 352|1440

run_lstm(Xtr, Xte, ytr, yte, max_features, max_features,
         out_size,embedding_size, hidden_size, batch_size, epochs, verbose, max_of_sentences)



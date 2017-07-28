# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import tensorflow as tf
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.python.framework import ops
import tflearn
#from tensorflow.python.ops.rnn_cell import MultiRNNCell, BasicLSTMCell, GRUCell
from tensorflow.python.ops import rnn
import numpy as np
from numpy import float32
import os
import re
import gc

# https://github.com/dhwajraj/NER-RNN/blob/master/NerMulti.py
from tflearn import GRUCell

EMBEDDING_SIZE = 311  # (300 for word2vec embeddings and 11 for extra features (POS,CHUNK,CAP))
MAX_DOCUMENT_LENGTH = 30
MAX_WORD_LENGTH = 15
num_classes = 5
BASE_DIR = "ner"  # path to coNLL data set


def pos(tag):
    onehot = np.zeros(5)
    if tag == 'NN' or tag == 'NNS':
        onehot[0] = 1
    elif tag == 'FW':
        onehot[1] = 1
    elif tag == 'NNP' or tag == 'NNPS':
        onehot[2] = 1
    elif 'VB' in tag:
        onehot[3] = 1
    else:
        onehot[4] = 1

    return onehot

def chunk(tag):
    onehot = np.zeros(5)
    if 'NP' in tag:
        onehot[0] = 1
    elif 'VP' in tag:
        onehot[1] = 1
    elif 'PP' in tag:
        onehot[2] = 1
    elif tag == 'O':
        onehot[3] = 1
    else:
        onehot[4] = 1

    return onehot

def capital(word):
    if ord(word[0]) >= 'A' and ord(word[0]) <= 'Z':
        return np.asarray([1])
    else:
        return np.asarray([0])

pre_emb = dict()
for line in open('/Users/esteves/Github/horus-models/data/dataset/myvectors.txt'):
    l = line.strip().split()
    w = l[0]
    arr = l[1:]
    pre_emb[w] = arr

print("PRE_EMB size", len(pre_emb))

def getEmb(w):
    randV = np.random.uniform(-0.25, 0.25, EMBEDDING_SIZE - 11)
    s = re.sub('[^0-9a-zA-Z]+', '', w)
    arr = []
    if w == "~#~":
        arr = [0 for _ in range(EMBEDDING_SIZE)]
    elif w in pre_emb:
        arr = pre_emb[w]
    elif w.lower() in pre_emb:
        arr = pre_emb[w.lower()]
    elif s in pre_emb:
        arr = pre_emb[s]
    elif s.isdigit():
        arr = pre_emb["1"]

    if len(arr) > 0:
        return np.asarray(arr)
    return randV

def get_input(FILE_NAME):
    word = []
    tag = []

    sentence = []
    sentence_tag = []

    # get max words in sentence
    max_sentence_length = MAX_DOCUMENT_LENGTH  # findMaxLenght(FILE_NAME)
    sentence_length = 0

    print("max sentence size is : " + str(max_sentence_length))

    for line in open(FILE_NAME):
        if line in ['\n', '\r\n']:
            # print("aa"+str(sentence_length) )
            for _ in range(max_sentence_length - sentence_length):
                tag.append(np.asarray([0, 0, 0, 0, 0]))
                getEmb("~#~")
                temp = np.zeros(11)
                word.append(temp)

            sentence.append(word)
            # print(len(word))
            sentence_tag.append(np.asarray(tag))

            sentence_length = 0
            word = []
            tag = []


        else:
            assert (len(line.split()) == 4)
            if sentence_length >= max_sentence_length:
                continue
            sentence_length += 1
            temp = getEmb(line.split()[0])
            temp = np.append(temp, pos(line.split()[1]))  # adding pos embeddings
            temp = np.append(temp, chunk(line.split()[2]))  # adding chunk embeddings
            temp = np.append(temp, capital(line.split()[0]))  # adding capital embedding
            word.append(temp)
            t = line.split()[3]

            # Five classes 0-None,1-Person,2-Location,3-Organisation,4-Misc

            if t.endswith('O'):
                tag.append(np.asarray([1, 0, 0, 0, 0]))
            elif t.endswith('PER'):
                tag.append(np.asarray([0, 1, 0, 0, 0]))
            elif t.endswith('LOC'):
                tag.append(np.asarray([0, 0, 1, 0, 0]))
            elif t.endswith('ORG'):
                tag.append(np.asarray([0, 0, 0, 1, 0]))
            elif t.endswith('MISC'):
                tag.append(np.asarray([0, 0, 0, 0, 1]))
            else:
                print("error in input" + str(t))

    assert (len(sentence) == len(sentence_tag))
    return np.asarray(sentence), sentence_tag

def cost(prediction, target):
    target = tf.reshape(target, [-1, MAX_DOCUMENT_LENGTH, num_classes])
    prediction = tf.reshape(prediction, [-1, MAX_DOCUMENT_LENGTH, num_classes])
    cross_entropy = target * tf.log(prediction)
    cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
    mask = tf.sign(tf.reduce_max(tf.abs(target), reduction_indices=2))
    cross_entropy *= mask
    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
    cross_entropy /= tf.cast(length(target), tf.float32)
    return tf.reduce_mean(cross_entropy)

def length(target):
    used = tf.sign(tf.reduce_max(tf.abs(target), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

def f1(prediction, target):  # not tensors but result values
    target = np.reshape(target, (-1, MAX_DOCUMENT_LENGTH, num_classes))
    prediction = np.reshape(prediction, (-1, MAX_DOCUMENT_LENGTH, num_classes))

    tp = np.asarray([0] * (num_classes + 2))
    fp = np.asarray([0] * (num_classes + 2))
    fn = np.asarray([0] * (num_classes + 2))

    target = np.argmax(target, 2)
    prediction = np.argmax(prediction, 2)

    for i in range(len(target)):
        for j in range(MAX_DOCUMENT_LENGTH):
            if target[i][j] == prediction[i][j]:
                tp[target[i][j]] += 1
            else:
                fp[target[i][j]] += 1
                fn[prediction[i][j]] += 1

    NON_NAMED_ENTITY = 0
    for i in range(num_classes):
        if i != NON_NAMED_ENTITY:
            tp[5] += tp[i]
            fp[5] += fp[i]
            fn[5] += fn[i]
        else:
            tp[6] += tp[i]
            fp[6] += fp[i]
            fn[6] += fn[i]

    precision = []
    recall = []
    fscore = []
    for i in range(num_classes + 2):
        precision.append(tp[i] * 1.0 / (tp[i] + fp[i]))
        recall.append(tp[i] * 1.0 / (tp[i] + fn[i]))
        fscore.append(2.0 * precision[i] * recall[i] / (precision[i] + recall[i]))

    print("precision = ", precision)
    print("recall = ", recall)
    print("f1score = ", fscore)
    efs = fscore[5]
    print("Entity fscore :", efs)
    del precision
    del recall
    del fscore
    return efs

X_train, y_train = get_input("/Users/esteves/Github/horus-models/data/dataset/coNLL2003/coNLL2003.eng.train")
X_test, y_test = get_input("/Users/esteves/Github/horus-models/data/dataset/coNLL2003/coNLL2003.eng.testa")
X_val, y_val = get_input("/Users/esteves/Github/horus-models/data/dataset/coNLL2003/coNLL2003.eng.testb")

y_train = np.asarray(y_train).astype(int).reshape(len(y_train), -1)
print(y_train.shape)

y_test = np.asarray(y_test).astype(int).reshape(len(y_test), -1)
y_val = np.asarray(y_val).astype(int).reshape(len(y_val), -1)
n_words = len(pre_emb)

del pre_emb

print("Network building")



net = tflearn.input_data([None, MAX_DOCUMENT_LENGTH, EMBEDDING_SIZE])
#net = tflearn.embedding(net, input_dim=n_words, output_dim=EMBEDDING_SIZE,trainable=True, name="EmbeddingLayer")
net = rnn.bidirectional_rnn(MultiRNNCell([GRUCell(256)]*3), MultiRNNCell([GRUCell(256)]*3), tf.unpack(tf.transpose(net, perm=[1, 0, 2])), dtype=tf.float32)  #256=num_hidden, 3=num_layers
net = tflearn.dropout(net[0],0.5)
net = tf.transpose(tf.pack(net), perm=[1, 0, 2])

net = tflearn.fully_connected(net, MAX_DOCUMENT_LENGTH*num_classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam',loss=cost)

model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=0)
print("Training")

# embeddingWeights = tflearn.get_layer_variables_by_name('EmbeddingLayer')[0]
# model.set_weights(embeddingWeights, initW)
gc.collect()
while True:
    with ops.get_default_graph().as_default():
        model.fit(X_train, y_train,n_epoch=1,validation_set=(X_test,y_test), show_metric=False, batch_size=200)
        predY=np.asarray(model.predict(X_val))
        f1(predY,y_val)
        del predY
        gc.collect()
    #print(classification_report(y_val, y_pred) )

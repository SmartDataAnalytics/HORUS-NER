import argparse
import os

import time

import gc
import copy
import keras
import sklearn
from keras.utils import to_categorical
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from src.util.definitions import encoder_le1_name, dict_exp_configurations, SET_MASK, schemaindex2label, NER_DATASETS_TRAIN_DEV, NER_DATASETS_TEST, optim_clf_rf, optim_clf_crf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import matplotlib.pyplot as plt
import sklearn_crfsuite
from sklearn import ensemble
from sklearn.cross_validation import train_test_split, KFold
from sklearn_crfsuite import metrics
from keras_contrib.layers import CRF


plt.style.use('ggplot')
from src.config import HorusConfig
from src.util import definitions
from sklearn.externals import joblib
import scipy.stats
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.grid_search import RandomizedSearchCV
import numpy as np
from keras.engine import InputLayer
from keras.models import Sequential, Model, Input
from keras.layers.core import Activation
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Merge, Bidirectional, Concatenate, Dropout
from nltk.corpus import stopwords
from nltk import LancasterStemmer, re, WordNetLemmatizer
import pandas as pd
import cPickle as pickle
# import pickle
import multiprocessing
from functools import partial
from contextlib import contextmanager
from nltk.stem.snowball import SnowballStemmer

"""
==========================================================
Experiments: 
    NER models X NER + HORUS features
==========================================================
Within this experiments we show the performance of standard
NER algorithms (Stanford NER and NLTK NER) with and without
using HORUS as features

input: 
- horus matrix file: see README -> https://github.com/diegoesteves/horus-ner

output:
- performance measures
"""
config = HorusConfig()
X, Y = [], []
ds_test_size = 0.20
stemmer = SnowballStemmer('english')
stop = set(stopwords.words('english'))
enc_le1 = joblib.load(config.dir_encoders + definitions.encoder_le1_name)
enc_le2 = joblib.load(config.dir_encoders + definitions.encoder_le2_name)
enc_word = joblib.load(config.dir_encoders + definitions.encoder_int_words_name)
enc_lemma = joblib.load(config.dir_encoders + definitions.encoder_int_lemma_name)
enc_stem = joblib.load(config.dir_encoders + definitions.encoder_int_stem_name)

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def encode(x, n):
    result = np.zeros(n)
    result[x] = 1
    return result
#keras.utils.np_utils.to_categorical or sparse_categorical_crossentropy

def load_dumps_in_memory((_file, ds, _set_name)):
    config.logger.info('loading dump file [%s]: %s' % (ds, _file))
    f = open(_file, 'rb')
    dump = pickle.load(f)
    f.close()
    return (_set_name, dump)

def encode_labels(max_sentence, y1, y2):
    #import copy
    #X_enc = copy.copy(X)
    #all_text = [c[index_word] for x in X for c in x]
    ##all_text.extend([c[1] for x in X for c in x])

    #words = list(set(all_text))  # distinct tokens
    #word2ind = {word: index for index, word in enumerate(words)}  # indexes of words
    #ind2word = {index: word for index, word in enumerate(words)}
    ##labels = list(set([c for x in y for c in x]))
    #label2ind =
    #ind2label = definitions.PLOMNone_index2label
    #print('Vocabulary size:', len(word2ind), len(label2ind))

    #print('min sentence / max sentence: ', min(lengths), maxlen)
    #for i in range(len(X_enc)):
    #    for t in range(len(X_enc[i])):
    #        w = enc_word.transform([str(X_enc[i].iloc[t][index_word])])
    #       X_enc[i].set_value(index=t, col=index_word, value=w[0])
    #        #X_enc[i].iloc[t][index_word] = w[0]

       ## X_enc = [[[word2ind[c[0]], word2ind[c[1]], c[2], c[3], c[4], c[5], c[6], c[7], c[8],
       ##            c[9], c[10], c[11], c[12], c[13], c[14], c[15], c[16], c[17], c[18]] for c in x] for x in Xclean]

    max_label = max(definitions.PLOMNone_label2index.values()) + 1
    y_enc1 = [[0] * (max_sentence - len(ey)) + [definitions.PLOMNone_label2index[c] for c in ey] for ey in y1]
    y_enc1 = [[encode(c, max_label) for c in ey] for ey in y_enc1]

    y_enc2 = [[0] * (max_sentence - len(ey)) + [definitions.PLOMNone_label2index[c] for c in ey] for ey in y2]
    y_enc2 = [[encode(c, max_label) for c in ey] for ey in y_enc2]

    #max_features = len(word2ind)




    return y_enc1, y_enc2

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

def run_bisltm(X1_word, X1_feat, Y1, one_hot_encode_y, out_file, ds1_label,
               f_key, line, random_state = None, random_state_i=None,
               X2_word=None, X2_feat=None, Y2=None, ds2_label=None):

    embedding_size = 128
    hidden_size = 32
    batch_size = 64
    epochs = 50
    verbose = 0

    import itertools
    try:

        X1_feat_temp = copy.deepcopy(X1_feat)

        assert ((X2_word is not None and X2_feat is not None and Y2 is not None) or
                (X2_word is None and X2_feat is None and Y2 is None))

        assert len(X1_feat_temp) == len(X1_word) == len(Y1)

        cross_val = False
        if X2_word is None:
            cross_val = True
            ds2_label = ds1_label


        # excluding word index
        if 88 in X1_feat_temp[0].keys():
            for df in X1_feat_temp:
                del df[88]
        lengths1 = [len(x) for x in X1_feat_temp]
        max_len1 = max(lengths1)
        out_size = len(definitions.PLOMNone_label2index) + 1
        y1_idx = [[definitions.PLOMNone_label2index[y] for y in s] for s in Y1]


        max_len2 = 0
        if cross_val\
                is False:
            if 88 in X2_feat[0].keys():
                for df in X2_feat:
                    del df[88]
            lengths2 = [len(x) for x in X2_feat]
            max_len2 = max(lengths2)
            y2_idx = [[definitions.PLOMNone_label2index[y] for y in s] for s in Y2]

        max_len = max(max_len1, max_len2)
        # n_features = len(X1_sentence_idxc[0].columns)
        # n_words = len(enc_word.classes_)
        y1_idx_pad = pad_sequences(sequences=y1_idx, maxlen=max_len, padding='post', value=0)  # value=definitions.PLOMNone_label2index['O']
        y1_idx_pad_enc = [one_hot_encode_y[y] for y in y1_idx_pad]
        X1_feat_pad = pad_sequences(sequences=[x.values.tolist() for x in X1_feat_temp], maxlen=max_len, padding='post', value=0)
        # word embedding layer - idx word = 85 (word.lemma)
        seq_words_X1 = [S[85].tolist() for S in X1_word]
        flatt1 = list(itertools.chain(*seq_words_X1))
        flatt1 = list(set(flatt1))
        flatt1.append("ENDPAD")
        len_words_X1 = len(flatt1)

        flatt2 = []
        len_words_X2 = 0
        if cross_val is False:
            y2_idx_pad = pad_sequences(sequences=y2_idx, maxlen=max_len, padding='post', value=0)
            y2_idx_pad_enc = [one_hot_encode_y[y] for y in y2_idx_pad]

            X2_feat_pad = pad_sequences(sequences=[x.values.tolist() for x in X2_feat], maxlen=max_len, padding='post', value=0)
            seq_words_X2 = [S[85].tolist() for S in X2_word]
            flatt2 = list(itertools.chain(*seq_words_X2))
            flatt2 = list(set(flatt2))
            flatt2.append("ENDPAD")
            len_words_X2 = len(flatt2)

        flatt = flatt1 + flatt2
        len_words_X = len_words_X1 + len_words_X2

        word2idx = {w: i + 1 for i, w in enumerate(flatt)}

        X1_sentence_only_words = [[word2idx[w] for w in s] for s in seq_words_X1]
        X1_sentence_only_words_pad = pad_sequences(sequences=X1_sentence_only_words, maxlen=max_len, padding='post', value=0)

        if cross_val is False:
            X2_sentence_only_words = [[word2idx[w] for w in s] for s in seq_words_X2]
            X2_sentence_only_words_pad = pad_sequences(sequences=X2_sentence_only_words, maxlen=max_len, padding='post', value=0)

        if cross_val is True:
            # words split
            Xtr, Xte, ytr, yte = train_test_split(X1_sentence_only_words_pad, y1_idx_pad_enc, test_size=ds_test_size, random_state=random_state)

            # feat split
            Xtr_h, Xte_h, ytr, yte = train_test_split(X1_feat_pad, y1_idx_pad_enc, test_size=ds_test_size, random_state=random_state)
        else:
            Xtr = X1_sentence_only_words_pad
            Xtr_h = X1_feat_pad
            ytr = y1_idx_pad_enc

            Xte = X2_sentence_only_words_pad
            Xte_h = X2_feat_pad
            yte = y2_idx_pad_enc


        '''
        word features
        '''
        input1 = Input(shape=(max_len,))
        word_emb = Embedding(input_dim=len_words_X + 1, output_dim=20,
                             input_length=max_len,
                             mask_zero=True)(input1)
        # model = Bidirectional(LSTM(units=50, return_sequences=True,
        #                           recurrent_dropout=0.1))(word_emb)
        # model = TimeDistributed(Dense(50, activation='relu'))(model)
        # crf = CRF(out_size)
        # out = crf(model)
        # model = Model(input1, out)
        # model.compile(optimizer='rmsprop', loss=crf.loss_function,
        #              metrics=[crf.accuracy])
        # print(model.summary())

        # hist = model.fit(Xtr, np.array(ytr), batch_size=64, epochs=50,
        #                 validation_split=0.2, verbose=0)
        # from seqeval.metrics import precision_score, recall_score, f1_score, \
        #    classification_report
        # test_pred = model.predict(Xte, verbose=1)

        # pred_labels = pred2label(test_pred)
        # test_labels = pred2label(yte)


        # _ypr_nn = np.array([tag for row in pred_labels for tag in row])
        # _yte_nn = np.array([tag for row in test_labels for tag in row])
        # P, R, F, S = sklearn.metrics.precision_recall_fscore_support(_yte_nn, _ypr_nn,
        #                                                             labels=definitions.PLOM_index2label.values())

        # print(sklearn.metrics.classification_report(_yte_nn, _ypr_nn,
        #                                            labels=definitions.PLOM_index2label.values(),
        #                                            digits=3))

        '''
        horus features
        '''
        input2 = Input(shape=(max_len,))
        horus_emb = Embedding(input_dim=100000, output_dim=20,
                              input_length=max_len,
                              mask_zero=True)(input2)

        concat_layer = keras.layers.concatenate([word_emb, horus_emb])

        model = Bidirectional(LSTM(units=50, return_sequences=True,
                                   recurrent_dropout=0.1))(concat_layer)
        model = Dense(150, activation='relu')(model)
        model = Dropout(0.1, noise_shape=None, seed=None)(model)
        model = Dense(150, activation='relu')(model)
        # model = TimeDistributed(Dense(50, activation='relu'))(model)
        crf = CRF(out_size)
        out = crf(model)

        model = Model(inputs=[input1, input2], outputs=[out])
        # model = Model(inputs=input2, outputs=out)
        model.compile(optimizer='rmsprop', loss=crf.loss_function,
                      metrics=[crf.accuracy])
        # print(model.summary())

        hist = model.fit([Xtr, Xtr_h[:, :, 0]], np.array(ytr), batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=verbose)
        from seqeval.metrics import precision_score, recall_score, f1_score, \
            classification_report
        test_pred = model.predict([Xte, Xte_h[:, :, 0]], verbose=verbose)

        pred_labels = pred2label(test_pred)
        test_labels = pred2label(yte)
        # fone = f1_score(test_labels, pred_labels)
        # print(classification_report(test_labels, pred_labels, digits=3))

        _ypr_nn = np.array([tag for row in pred_labels for tag in row])
        _yte_nn = np.array([tag for row in test_labels for tag in row])
        P, R, F, S = sklearn.metrics.precision_recall_fscore_support(_yte_nn, _ypr_nn,
                                                                     labels=definitions.PLOM_index2label.values())

        # print(sklearn.metrics.classification_report(_yte_nn, _ypr_nn,
        #            labels=definitions.PLOM_index2label.values(), digits=3))

        cross_str = False
        run = '0'
        if cross_val:
            cross_str = True
            run = str(random_state_i + 1)
            for k in range(len(P)):
                out_file.write(line % (
                    'True', str(f_key), str(random_state_i + 1), definitions.PLOM_index2label.get(k + 1),
                    P[k], R[k],
                    F[k],
                    str(S[k]), 'BiLSTM+CRF', ds1_label, ds2_label, 'NER'))

        # average

        P_avg, R_avg, F_avg, S_avg = sklearn.metrics.precision_recall_fscore_support(
            _yte_nn, _ypr_nn, labels=definitions.PLOM_index2label.values(),
            average='weighted')
        out_file.write(line % (
            cross_str, str(f_key), run, 'average', P_avg, R_avg, F_avg, 0, 'BiLSTM+CRF', ds1_label, ds2_label, 'NER'))

        out_file.flush()
    except Exception as e:
        config.logger.error(repr(e))
        raise

def run_lstm(Xtr, Xte, ytr, yte, max_features, max_features2, out_size, embedding_size, hidden_size, batch_size,
             epochs=50, verbose=0, maxsent=0):

    print('Training and testing tensor shapes:', Xtr.shape, Xte.shape, ytr.shape, yte.shape)

    mf = max(max_features, max_features2)

    model1 = Sequential()
    model1.add(Embedding(input_dim=mf, output_dim=embedding_size, input_length=maxsent, dropout=0.2)) # mask_zero=True
    model1.add(LSTM(hidden_size, dropout = 0.2)) #return_sequences=True, input_shape=(maxsent, Xtr.shape[2] - 1)
    model1.add(Dense(1, activation='softmax'))
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model1.summary())
    model1.fit(Xtr, ytr, epochs=3, batch_size=64)
    scores = model1.evaluate(Xte, yte, verbose=0)

    model2 = Sequential()
    model2.add(InputLayer(input_shape=(maxsent, Xtr.shape[2] - 1)))

    model = Sequential()
    model.add(Merge([model1, model2], mode='concat'))



    model.add(TimeDistributed(Dense(out_size)))
    model.add(Activation('softmax'))


    #print(model.summary())
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

def pred2label(pred):
    out = []
    try:
        for pred_i in pred:
            out_i = []
            for p in pred_i:
                p_i = np.argmax(p)
                if p_i == 0:
                    out_i.append('O')
                else:
                    out_i.append(definitions.PLOMNone_index2label[p_i])
            out.append(out_i)
    except Exception as e:
        config.logger.error(repr(e))
    return out

from nltk.tag.stanford import StanfordNERTagger
import nltk

def benchmark(experiment_folder, runCRF = False, runRF = False, runLSTM = False,
              runSTANFORD_NER = False, optimization = False, keys = []):

    config.logger.info('models: CRF=%s, DT=%s, LSTM=%s, Stanford=%s' % (str(runCRF), str(runRF), str(runLSTM), str(runSTANFORD_NER)))
    experiment_folder+='/'

    #sorted_labels = definitions.KLASSES.copy()
    #del sorted_labels[4]
    sorted_labels={'PER': 'PER', 'ORG': 'ORG', 'LOC': 'LOC'}
    r = [42, 19, 10] #3 folds cross validation
    # hyper-parameters

    '''
    classifiers
    '''
    _crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.088, c2=0.002, max_iterations=100, all_possible_transitions=True)
    _crf2 = sklearn_crfsuite.CRF(algorithm='pa', all_possible_transitions=True)
    _rf50 = ensemble.RandomForestClassifier(n_estimators=50)
    _dt = DecisionTreeClassifier()
    '''
    end classifiers
    '''
    # 0 should be different to idx of 'O' IMO! so I keep 0 for data padding.
    temp = [0]
    temp.extend(definitions.PLOMNone_index2label.keys())
    one_hot_encode_y = to_categorical(temp)
    enc_word = joblib.load(config.dir_encoders + definitions.encoder_int_words_name)
    #_meta = MEX('HORUS_EMNLP', _label, 'meta and multi-level machine learning for NLP')


    config.logger.info('running benchmark...')
    # benchmark starts
    name='metadata_'
    if runCRF: name+='crf_'
    if runRF: name+= 'trees_'
    if runLSTM: name+='lstm_'
    if runSTANFORD_NER: name+='stanford_'
    assert name != 'metadata_'

    name += ''.join(map(str,(keys)))
    name +='.txt'
    out_file = open(config.dir_output + name, 'w+')
    out_file.write(definitions.header)
    for f_key in keys:
        config.logger.info('loading dumps for configuration: ' + str(f_key))
        try:
            for ds1 in NER_DATASETS_TRAIN_DEV:
                horus_m4_name_ds1 = ds1[0][0]
                horus_m4_path_ds1 = ds1[0][1] + ds1[0][2].replace('.horusx', '')
                dump_name = SET_MASK % (horus_m4_name_ds1, str(f_key))
                dump_full_path_ds1_sentence = os.path.dirname(os.path.realpath(horus_m4_path_ds1)) + '/' + dump_name.replace('.pkl', '.sentence.pkl')
                dump_full_path_ds1_sentence_idx = os.path.dirname(os.path.realpath(horus_m4_path_ds1)) + '/' + dump_name.replace('.pkl', '.sentence.idx.pkl')
                dump_full_path_ds1_token = os.path.dirname(os.path.realpath(horus_m4_path_ds1)) + '/' + dump_name.replace('.pkl', '.token.pkl')
                dump_full_path_ds1_crf = os.path.dirname(os.path.realpath(horus_m4_path_ds1)) + '/' + dump_name.replace('.pkl', '.crf.pkl')

                if not os.path.isfile(dump_full_path_ds1_sentence):
                    config.logger.info(dump_full_path_ds1_sentence)
                    config.logger.error(' -- configuration file does not exist! check its creation')
                    raise Exception
                else:
                    config.logger.debug('(ds1) loading: ' + dump_name + ' dump files')
                    config.logger.debug(' - ' + dump_full_path_ds1_sentence)
                    config.logger.debug(' - ' + dump_full_path_ds1_sentence_idx)
                    config.logger.debug(' - ' + dump_full_path_ds1_token)
                    config.logger.debug(' - ' + dump_full_path_ds1_crf)
                    config.logger.debug('--------------------------------------------')

                    with open(dump_full_path_ds1_sentence, 'rb') as input:
                        file_name, f_key, X1_sentence, Y1_sentence = pickle.load(input)
                        temp = [['O' if item == 'MISC' else item for item in lst] for lst in Y1_sentence]
                        Y1_sentence = temp

                    with open(dump_full_path_ds1_sentence_idx, 'rb') as input:
                        file_name, f_key, X1_sentence_idx, Y1_sentence_idx = pickle.load(input)

                    with open(dump_full_path_ds1_token, 'rb') as input:
                        file_name, f_key, X1_token, Y1_token = pickle.load(input)
                        temp = [definitions.PLOMNone_label2index['O'] if y==definitions.PLOMNone_label2index['MISC'] else y for y in Y1_token]
                        Y1_token = temp

                    with open(dump_full_path_ds1_crf, 'rb') as input:
                        file_name, f_key, X1_crf, Y1_crf = pickle.load(input)
                        Y1_crf = Y1_sentence

                    config.logger.debug('checking dump files (ds1)')
                    assert (len(X1_sentence) == len(Y1_sentence) == len(X1_sentence_idx) == len(Y1_sentence_idx))
                    config.logger.debug('ok!')

                    # same dataset (cross-validation)
                    # ---------------------------------------------------------- META ----------------------------------------------------------
                    # _conf = MEXConfiguration(id=len(_meta.configurations)+1, horus_enabled=int(horus_feat),
                    #                                dataset_train=ds1[0], dataset_test=ds1[0], dataset_validation=None, features=None, cross_validation=1)
                    # --------------------------------------------------------------------------------------------------------------------------
                    if X1_token.empty is True:
                        raise Exception('X1_token is empty!')

                    if not optimization:
                        for d in range(len(r)):
                            if runRF is True:
                                Xtr, Xte, ytr, yte = train_test_split(X1_token, Y1_token, test_size=ds_test_size,
                                                                      random_state=r[d])
                                '''
                                Random Forest
                                '''
                                m = _rf50.fit(np.array(Xtr).astype(float), np.array(ytr).astype(int))
                                # print(m.feature_importances_)
                                ypr = m.predict(np.array(Xte).astype(float))
                                # print(skmetrics.classification_report(np.array(yte).astype(int), np.array(ypr).astype(int), labels=definitions.PLO_KLASSES.keys(), target_names=definitions.PLO_KLASSES.values(), digits=3))
                                P, R, F, S = sklearn.metrics.precision_recall_fscore_support(np.array(yte).astype(int),
                                                                                             np.array(ypr).astype(int),
                                                                                             labels=definitions.PLOM_index2label.keys())
                                for k in range(len(P)):
                                    out_file.write(definitions.line % ('True', str(f_key), str(d + 1), definitions.PLOM_index2label.get(k + 1),
                                                           P[k], R[k], F[k], str(S[k]), 'RF50', horus_m4_name_ds1, horus_m4_name_ds1, 'NER'))

                                # average
                                P_avg, R_avg, F_avg, S_avg = sklearn.metrics.precision_recall_fscore_support(
                                    np.array(yte).astype(int), np.array(ypr).astype(int),
                                    labels=definitions.PLOM_index2label.keys(),
                                    average='weighted')
                                out_file.write(definitions.line % ('True', str(f_key), str(d + 1), 'average', P_avg, R_avg, F_avg, 0, 'RF50', horus_m4_name_ds1, horus_m4_name_ds1, 'NER'))

                                # entity detection only
                                ypr_bin = [1 if x in definitions.PLOM_index2label.keys() else 0 for x in ypr]
                                y2_bin = [1 if x in definitions.PLOM_index2label.keys() else 0 for x in yte]
                                P, R, F, S = sklearn.metrics.precision_recall_fscore_support(y2_bin, ypr_bin)
                                for k in range(len(P)):
                                    out_file.write(definitions.line % ('True', str(f_key), str(d + 1), k, P[k], R[k], F[k], str(S[k]),
                                                           'RF50', horus_m4_name_ds1, horus_m4_name_ds1, 'NED'))

                                '''
                                Decision Trees
                                '''
                                m = _dt.fit(np.array(Xtr).astype(float), np.array(ytr).astype(int))
                                # print(m.feature_importances_)
                                ypr = m.predict(np.array(Xte).astype(float))
                                # print(skmetrics.classification_report(np.array(yte).astype(int), np.array(ypr).astype(int), labels=definitions.PLO_KLASSES.keys(), target_names=definitions.PLO_KLASSES.values(), digits=3))
                                P, R, F, S = sklearn.metrics.precision_recall_fscore_support(np.array(yte).astype(int),
                                                                                             np.array(ypr).astype(int),
                                                                                             labels=definitions.PLOM_index2label.keys())
                                for k in range(len(P)):
                                    out_file.write(definitions.line % ('True', str(f_key), str(d + 1), definitions.PLOM_index2label.get(k + 1),
                                                           P[k], R[k], F[k], str(S[k]), 'DT', horus_m4_name_ds1, horus_m4_name_ds1, 'NER'))

                                # average
                                P_avg, R_avg, F_avg, S_avg = sklearn.metrics.precision_recall_fscore_support(
                                    np.array(yte).astype(int), np.array(ypr).astype(int),
                                    labels=definitions.PLOM_index2label.keys(),
                                    average='weighted')
                                out_file.write(definitions.line % ('True', str(f_key), str(d + 1), 'average', P_avg, R_avg, F_avg, 0, 'DT', horus_m4_name_ds1, horus_m4_name_ds1, 'NER'))

                                # entity detection only
                                ypr_bin = [1 if x in definitions.PLOM_index2label.keys() else 0 for x in ypr]
                                y2_bin = [1 if x in definitions.PLOM_index2label.keys() else 0 for x in yte]
                                P, R, F, S = sklearn.metrics.precision_recall_fscore_support(y2_bin, ypr_bin)
                                for k in range(len(P)):
                                    out_file.write(definitions.line % ('True', str(f_key), str(d + 1), k, P[k], R[k], F[k], str(S[k]),
                                                           'DT', horus_m4_name_ds1, horus_m4_name_ds1, 'NED'))

                                # ---------------------------------------------------------- META ----------------------------------------------------------
                                # _ex = MEXExecution(id=len(_conf.executions) + 1, model='', alg='RF50', phase='test', random_state=r[d])
                                # P, R, F, S = sklearn.metrics.precision_recall_fscore_support(yte, ypr,
                                #                                                             labels=sorted_labels.keys(),
                                #                                                             average=None)
                                # for k in sorted_labels.keys():
                                #    _ex.add_performance(MEXPerformance(k, P[k], R[k], F[k], 0.0, S[k]))
                                # _conf.add_execution(_ex)
                                # _meta.add_configuration(_conf)
                                # --------------------------------------------------------------------------------------------------------------------------

                            if runCRF is True:
                                Xtr, Xte, ytr, yte = \
                                    train_test_split(X1_crf, Y1_sentence, test_size=ds_test_size, random_state=r[d])

                                m = _crf.fit(Xtr, ytr)
                                ypr = m.predict(Xte)
                                _ypr = np.array([tag for row in ypr for tag in row])
                                _yte = np.array([tag for row in yte for tag in row])
                                # print(metrics.flat_classification_report(yte, ypr, labels=sorted_labels.keys(), target_names=sorted_labels.values(), digits=3))
                                P, R, F, S = sklearn.metrics.precision_recall_fscore_support(_yte, _ypr,
                                                                                             labels=definitions.PLOM_index2label.values())
                                for k in range(len(P)):
                                    out_file.write(definitions.line % (
                                        'True', str(f_key), str(d + 1), definitions.PLOM_index2label.get(k + 1), P[k], R[k],
                                        F[k],
                                        str(S[k]), 'CRF', horus_m4_name_ds1, horus_m4_name_ds1, 'NER'))

                                # average
                                P_avg, R_avg, F_avg, S_avg = sklearn.metrics.precision_recall_fscore_support(_yte, _ypr, labels=definitions.PLOM_index2label.values(), average='weighted')
                                out_file.write(definitions.line % ('True', str(f_key), str(d + 1), 'average', P_avg, R_avg, F_avg, 0, 'CRF', horus_m4_name_ds1, horus_m4_name_ds1, 'NER'))

                                # entity detection only
                                ypr_bin = [1 if x in definitions.PLOM_index2label.values() else 0 for x in _ypr]
                                y2_bin = [1 if x in definitions.PLOM_index2label.values() else 0 for x in _yte]
                                P, R, F, S = sklearn.metrics.precision_recall_fscore_support(y2_bin, ypr_bin)
                                for k in range(len(P)):
                                    out_file.write(definitions.line % (
                                        'True', str(f_key), str(d + 1), k, P[k], R[k], F[k], str(S[k]), 'CRF', horus_m4_name_ds1, horus_m4_name_ds1, 'NED'))

                                m = _crf2.fit(Xtr, ytr)
                                ypr = m.predict(Xte)
                                _ypr = np.array([tag for row in ypr for tag in row])
                                _yte = np.array([tag for row in yte for tag in row])
                                P, R, F, S = sklearn.metrics.precision_recall_fscore_support(_yte, _ypr,
                                                                                             labels=definitions.PLOM_index2label.values())
                                for k in range(len(P)):
                                    out_file.write(definitions.line % (
                                        'True', str(f_key), str(d + 1), definitions.PLOM_index2label.get(k + 1), P[k], R[k],
                                        F[k],
                                        str(S[k]), 'CRF_PA', horus_m4_name_ds1, horus_m4_name_ds1, 'NER'))

                                # average
                                P_avg, R_avg, F_avg, S_avg = sklearn.metrics.precision_recall_fscore_support(
                                    _yte, _ypr, labels=definitions.PLOM_index2label.values(),
                                    average='weighted')
                                out_file.write(definitions.line % (
                                    'True', str(f_key), str(d + 1), 'average', P_avg, R_avg, F_avg, 0, 'CRF_PA',
                                    horus_m4_name_ds1, horus_m4_name_ds1, 'NER'))

                                # entity detection only
                                ypr_bin = [1 if x in definitions.PLOM_index2label.values() else 0 for x in _ypr]
                                y2_bin = [1 if x in definitions.PLOM_index2label.values() else 0 for x in _yte]
                                P, R, F, S = sklearn.metrics.precision_recall_fscore_support(y2_bin, ypr_bin)
                                for k in range(len(P)):
                                    out_file.write(definitions.line % (
                                        'True', str(f_key), str(d + 1), k, P[k], R[k], F[k], str(S[k]), 'CRF_PA', horus_m4_name_ds1, horus_m4_name_ds1, 'NED'))

                                # ---------------------------------------------------------- META ----------------------------------------------------------
                                # _ex = MEXExecution(id=len(_conf.executions)+1, model='', alg='CRF', phase='test', random_state=r[d])
                                # P, R, F, S = sklearn.metrics.precision_recall_fscore_support(yte, ypr, labels=sorted_labels.keys(), average=None)
                                # for k in sorted_labels.keys():
                                #    _ex.add_performance(MEXPerformance(k, P[k], R[k], F[k], 0.0, S[k]))
                                # _conf.add_execution(_ex)
                                # _meta.add_configuration(_conf)
                                # --------------------------------------------------------------------------------------------------------------------------

                            if runLSTM is True:
                                run_bisltm(X1_sentence, X1_sentence_idx, Y1_sentence, one_hot_encode_y,
                                           out_file, horus_m4_name_ds1, f_key, definitions.line, r[d], d)


                    for ds2 in NER_DATASETS_TEST:
                        horus_m4_name_ds2 = ds2[0]
                        horus_m4_path_ds2 = ds2[1] + ds2[2].replace('.horusx', '.horus4')
                        dump_name = SET_MASK % (horus_m4_name_ds2, str(f_key))
                        dump_full_path_ds2_sentence = os.path.dirname(os.path.realpath(horus_m4_path_ds2)) + '/' + dump_name.replace('.pkl', '.sentence.pkl')
                        dump_full_path_ds2_sentence_idx = os.path.dirname(os.path.realpath(horus_m4_path_ds2)) + '/' + dump_name.replace('.pkl', '.sentence.idx.pkl')
                        dump_full_path_ds2_token = os.path.dirname(os.path.realpath(horus_m4_path_ds2)) + '/' + dump_name.replace('.pkl', '.token.pkl')
                        dump_full_path_ds2_crf = os.path.dirname(os.path.realpath(horus_m4_path_ds2)) + '/' + dump_name.replace('.pkl', '.crf.pkl')

                        if not os.path.isfile(dump_full_path_ds2_sentence):
                            config.logger.info(dump_full_path_ds2_sentence)
                            config.logger.error(' -- configuration file does not exist! check its creation')
                            raise Exception
                        else:
                            config.logger.info('%s -> %s' % (horus_m4_name_ds1, horus_m4_name_ds2))
                            if horus_m4_name_ds1 == horus_m4_name_ds2:
                                X2_sentence, Y2_sentence = None, None
                                X2_sentence_idx, Y2_sentence_idx = None, None
                                X2_token, Y2_token = None, None
                                X2_crf, Y2_crf = None, None
                            else:
                                config.logger.debug('(ds2) loading: ' + dump_name + ' dump files')
                                config.logger.debug(' - ' + dump_full_path_ds2_sentence)
                                config.logger.debug(' - ' + dump_full_path_ds2_sentence_idx)
                                config.logger.debug(' - ' + dump_full_path_ds2_token)
                                config.logger.debug(' - ' + dump_full_path_ds2_crf)
                                config.logger.debug('--------------------------------------------')

                                with open(dump_full_path_ds2_sentence, 'rb') as input:
                                    file_name, f_key, X2_sentence, Y2_sentence = pickle.load(input)
                                    temp = [['O' if item == 'MISC' else item for item in lst] for lst in Y2_sentence]
                                    Y2_sentence = temp

                                with open(dump_full_path_ds2_sentence_idx, 'rb') as input:
                                    file_name, f_key, X2_sentence_idx, Y2_sentence_idx = pickle.load(input)

                                with open(dump_full_path_ds2_token, 'rb') as input:
                                    file_name, f_key, X2_token, Y2_token = pickle.load(input)
                                    temp = [definitions.PLOMNone_label2index['O'] if y == definitions.PLOMNone_label2index['MISC'] else y for y in Y2_token]
                                    Y2_token = temp

                                with open(dump_full_path_ds2_crf, 'rb') as input:
                                    file_name, f_key, X2_crf, Y2_crf = pickle.load(input)
                                    Y2_crf = Y2_sentence

                                config.logger.debug('checking dump files (X2)')
                                assert (len(X2_sentence) == len(Y2_sentence) == len(X2_sentence_idx) == len(Y2_sentence_idx))
                                assert (X2_token is not None and not X2_token.empty)
                                config.logger.debug('ok!')


                            # ---------------------------------------------------------- META ----------------------------------------------------------
                            # _conf = MEXConfiguration(id=len(_meta.configurations) + 1, horus_enabled=int(horus_feat),
                            #                         dataset_train=ds1[0], dataset_test=ds2[0] ,features=ds1[1], cross_validation=0)
                            # --------------------------------------------------------------------------------------------------------------------------

                            if optimization:
                                if runRF:
                                    optim_clf_rf.fit(X1_token, Y1_token)
                                    ypr = optim_clf_rf.best_estimator_.predict(X2_token)
                                    P, R, F, S = \
                                        sklearn.metrics.precision_recall_fscore_support(Y2_token, np.array(ypr).astype(int),
                                                                                        labels=definitions.PLOM_index2label.keys())
                                    for k in range(len(P)):
                                        out_file.write(definitions.line % ('False', str(f_key), '1', definitions.PLOM_index2label.get(k + 1),
                                                                           P[k], R[k], F[k], str(S[k]), 'RF_optim', horus_m4_name_ds1, horus_m4_name_ds2, 'NER'))

                                    # average
                                    P_avg, R_avg, F_avg, S_avg = sklearn.metrics.precision_recall_fscore_support(Y2_token, np.array(ypr).astype(int),
                                                                                                                 labels=definitions.PLOM_index2label.keys(), average='weighted')
                                    out_file.write(definitions.line % (
                                        'False', str(f_key), '0', 'average', P_avg, R_avg, F_avg, 0, 'RF_optim', horus_m4_name_ds1, horus_m4_name_ds2, 'NER'))
                                    out_file.flush()

                                if runCRF:
                                    optim_clf_crf.fit(X1_crf, Y1_sentence)
                                    ypr = optim_clf_crf.best_estimator_.predict(X2_crf)
                                    _ypr = np.array([tag for row in ypr for tag in row])
                                    _yte = np.array([tag for row in Y2_sentence for tag in row])
                                    P, R, F, S = sklearn.metrics.precision_recall_fscore_support(_yte, _ypr,
                                                                                                 labels=definitions.PLOM_index2label.values())
                                    for k in range(len(P)):
                                        out_file.write(definitions.line % (
                                            'False', str(f_key), '1', definitions.PLOM_index2label.get(k + 1), P[k], R[k], F[k], str(S[k]),
                                            'CRF_optim', horus_m4_name_ds1, horus_m4_name_ds2, 'NER'))

                                    # average
                                    P_avg, R_avg, F_avg, S_avg = sklearn.metrics.precision_recall_fscore_support(
                                        _yte, _ypr,
                                        labels=definitions.PLOM_index2label.values(), average='weighted')
                                    out_file.write(definitions.line % (
                                        'False', str(f_key), '0', 'average', P_avg, R_avg, F_avg, 0, 'CRF_optim',
                                        horus_m4_name_ds1, horus_m4_name_ds2, 'NER'))
                                    out_file.flush()
                            else:
                                if runRF is True:
                                    '''
                                    Random Forest
                                    '''
                                    m = _rf50.fit(X1_token, Y1_token)
                                    ypr = m.predict(X2_token)
                                    # print(skmetrics.classification_report(Y2_token , ypr, labels=PLO_KLASSES.keys(), target_names=PLO_KLASSES.values(), digits=3))
                                    P, R, F, S = \
                                        sklearn.metrics.precision_recall_fscore_support(Y2_token, np.array(ypr).astype(int),
                                                                                        labels=definitions.PLOM_index2label.keys())
                                    for k in range(len(P)):
                                        out_file.write(definitions.line % ('False', str(f_key), '1', definitions.PLOM_index2label.get(k + 1),
                                                               P[k], R[k], F[k], str(S[k]), 'RF50', horus_m4_name_ds1, horus_m4_name_ds2, 'NER'))

                                    # average
                                    P_avg, R_avg, F_avg, S_avg = sklearn.metrics.precision_recall_fscore_support(Y2_token, np.array(ypr).astype(int),
                                                                                                                 labels=definitions.PLOM_index2label.keys(), average='weighted')
                                    out_file.write(definitions.line % (
                                    'False', str(f_key), '0', 'average', P_avg, R_avg, F_avg, 0, 'RF50', horus_m4_name_ds1, horus_m4_name_ds2, 'NER'))

                                    # entity detection only
                                    ypr_bin = [1 if x in definitions.PLOM_index2label.keys() else 0 for x in ypr]
                                    y2_bin = [1 if x in definitions.PLOM_index2label.keys() else 0 for x in Y2_token]
                                    P, R, F, S = sklearn.metrics.precision_recall_fscore_support(y2_bin, ypr_bin)
                                    for k in range(len(P)):
                                        out_file.write(definitions.line % ('False', str(f_key), '1', k,
                                                               P[k], R[k], F[k], str(S[k]), 'RF50', horus_m4_name_ds1, horus_m4_name_ds2, 'NED'))

                                    '''
                                    Decision Trees
                                    '''
                                    m = _dt.fit(X1_token, Y1_token)
                                    ypr = m.predict(X2_token)
                                    P, R, F, S = \
                                        sklearn.metrics.precision_recall_fscore_support(Y2_token, np.array(ypr).astype(int),
                                                                                        labels=definitions.PLOM_index2label.keys())
                                    for k in range(len(P)):
                                        out_file.write(definitions.line % ('False', str(f_key), '1', definitions.PLOM_index2label.get(k + 1),
                                                               P[k], R[k], F[k], str(S[k]), 'DT', horus_m4_name_ds1, horus_m4_name_ds2, 'NER'))

                                    # average
                                    P_avg, R_avg, F_avg, S_avg = sklearn.metrics.precision_recall_fscore_support(Y2_token, np.array(ypr).astype(int),
                                                                                                                 labels=definitions.PLOM_index2label.keys(), average='weighted')
                                    out_file.write(definitions.line % (
                                        'False', str(f_key), '0', 'average', P_avg, R_avg, F_avg, 0, 'DT', horus_m4_name_ds1, horus_m4_name_ds2, 'NER'))

                                    # entity detection only
                                    ypr_bin = [1 if x in definitions.PLOM_index2label.keys() else 0 for x in ypr]
                                    y2_bin = [1 if x in definitions.PLOM_index2label.keys() else 0 for x in Y2_token]
                                    P, R, F, S = sklearn.metrics.precision_recall_fscore_support(y2_bin, ypr_bin)
                                    for k in range(len(P)):
                                        out_file.write(definitions.line % ('False', str(f_key), '1', k,
                                                               P[k], R[k], F[k], str(S[k]), 'DT', horus_m4_name_ds1, horus_m4_name_ds2, 'NED'))
                                    # ---------------------------------------------------------- META ----------------------------------------------------------
                                    # _ex = MEXExecution(id=len(_conf.executions) + 1, alg='RF50', phase='test', random_state=r[d])
                                    # P, R, F, S = sklearn.metrics.precision_recall_fscore_support(ds2[1][3] , ypr,
                                    #                                                             labels=sorted_labels.keys(),
                                    #                                                             average=None)
                                    # for k in sorted_labels.keys():
                                    #    _ex.add_performance(MEXPerformance(k, P[k], R[k], F[k], 0.0, S[k]))
                                    # _conf.add_execution(_ex)
                                    # _meta.add_configuration(_conf)
                                    # --------------------------------------------------------------------------------------------------------------------------

                                if runCRF is True:
                                    m = _crf.fit(X1_crf, Y1_sentence)
                                    ypr = m.predict(X2_crf)
                                    #print(metrics.flat_classification_report(Y2_sentence, ypr, labels=sorted_labels.keys(), target_names=sorted_labels.values(), digits=3))
                                    _ypr = np.array([tag for row in ypr for tag in row])
                                    _yte = np.array([tag for row in Y2_sentence for tag in row])
                                    P, R, F, S = sklearn.metrics.precision_recall_fscore_support(_yte, _ypr,
                                                                                                 labels=definitions.PLOM_index2label.values())
                                    for k in range(len(P)):
                                        out_file.write(definitions.line % (
                                        'False', str(f_key), '1', definitions.PLOM_index2label.get(k + 1), P[k], R[k], F[k], str(S[k]),
                                        'CRF', horus_m4_name_ds1, horus_m4_name_ds2, 'NER'))

                                    # average
                                    P_avg, R_avg, F_avg, S_avg = sklearn.metrics.precision_recall_fscore_support(
                                        _yte, _ypr,
                                        labels=definitions.PLOM_index2label.values(), average='weighted')
                                    out_file.write(definitions.line % (
                                        'False', str(f_key), '0', 'average', P_avg, R_avg, F_avg, 0, 'CRF',
                                        horus_m4_name_ds1, horus_m4_name_ds2, 'NER'))

                                    # entity detection only
                                    ypr_bin = [1 if x in definitions.PLOM_index2label.values() else 0 for x in _ypr]
                                    y2_bin = [1 if x in definitions.PLOM_index2label.values() else 0 for x in _yte]
                                    P, R, F, S = sklearn.metrics.precision_recall_fscore_support(y2_bin, ypr_bin)
                                    for k in range(len(P)):
                                        out_file.write(definitions.line % (
                                            'False', str(f_key), '1', k, P[k], R[k], F[k], str(S[k]), 'CRF', horus_m4_name_ds1, horus_m4_name_ds2, 'NED'))


                                    m = _crf2.fit(X1_crf, Y1_sentence)
                                    ypr = m.predict(X2_crf)
                                    _ypr = np.array([tag for row in ypr for tag in row])
                                    _yte = np.array([tag for row in Y2_sentence for tag in row])
                                    P, R, F, S = sklearn.metrics.precision_recall_fscore_support(_yte, _ypr,
                                                                                                 labels=definitions.PLOM_index2label.values())
                                    for k in range(len(P)):
                                        out_file.write(definitions.line % (
                                        'False', str(f_key), '1', definitions.PLOM_index2label.get(k + 1), P[k], R[k], F[k], str(S[k]),
                                        'CRF_PA', horus_m4_name_ds1, horus_m4_name_ds2, 'NER'))

                                    # average
                                    P_avg, R_avg, F_avg, S_avg = sklearn.metrics.precision_recall_fscore_support(
                                        _yte, _ypr,
                                        labels=definitions.PLOM_index2label.values(), average='weighted')
                                    out_file.write(definitions.line % (
                                        'False', str(f_key), '0', 'average', P_avg, R_avg, F_avg, 0, 'CRF-PA',
                                        horus_m4_name_ds1, horus_m4_name_ds2, 'NER'))

                                    # entity detection only
                                    ypr_bin = [1 if x in definitions.PLOM_index2label.values() else 0 for x in _ypr]

                                    P, R, F, S = sklearn.metrics.precision_recall_fscore_support(y2_bin, ypr_bin)
                                    for k in range(len(P)):
                                        out_file.write(definitions.line % (
                                            'False', str(f_key), '1', k, P[k], R[k], F[k], str(S[k]), 'CRF_PA', horus_m4_name_ds1, horus_m4_name_ds2, 'NED'))

                                if runLSTM is True:
                                    run_bisltm(X1_sentence, X1_sentence_idx, Y1_sentence, one_hot_encode_y,
                                               out_file, horus_m4_name_ds1, f_key, definitions.line, random_state=None,
                                               random_state_i=None, X2_word=X2_sentence, X2_feat=X2_sentence_idx,
                                               Y2=Y2_sentence, ds2_label=horus_m4_name_ds2)




                        out_file.flush()

        except Exception as e:
            config.logger.error(repr(e))
    out_file.close()
    #with open(_label + '.meta', 'wb') as handle:
    #    pickle.dump(_meta, handle, protocol=pickle.HIGHEST_PROTOCOL)

def benchmark_no_train():
    try:
        stanford_ner_tagger = StanfordNERTagger(config.model_stanford_filename_ner,
                                       config.model_stanford_path_jar_ner,
                                       encoding='utf8')

        out_file = open(config.dir_output + 'no_training.txt', 'w+')
        out_file.write(definitions.header)

        for ds1 in NER_DATASETS_TRAIN_DEV:
            label = ds1[0][0]
            path = ds1[0][1] + ds1[0][2].replace('.horusx', '')
            w = []
            y = []
            predicted = []
            real = []
            i=0
            with open(path, 'r') as fset:
                for rec in fset:
                    #if i>100:
                    #    break
                    if rec == '\n':
                        i+=1
                        if len(w) != len(y):
                            a=1
                        out = stanford_ner_tagger.tag(w)
                        out = definitions.tags_to_3muc_simple(out)
                        y = definitions.tags_to_3muc_simple(y)
                        assert len(out) == len(y)
                        predicted.extend(out)
                        real.extend(y)
                        w = []
                        y = []
                        out = []
                    else:
                        rec = rec.split('\t')
                        w.append(rec[0])
                        y.append(rec[1].replace('\n', ''))
            P, R, F, S = sklearn.metrics.precision_recall_fscore_support(np.array(real).astype(int),
                                                                         np.array(predicted).astype(int),
                                                                         labels=definitions.PLOM_index2label.keys())
            for k in range(len(P)):
                out_file.write(definitions.line % ('True', '0', '0', definitions.PLOM_index2label.get(k + 1),
                                       P[k], R[k], F[k], str(S[k]), 'stanford', label, label, 'NER'))
            out_file.flush()

    except:
        raise

def main():
    parser = argparse.ArgumentParser(
        description='Creates a benchmark pipeline for different classifiers /datasets comparing performance *with* '
                    'and *without* the HORUS features list',
        prog='benchmarking.py',
        usage='%(prog)s [options]',
        epilog='http://horus-ner.org')

    #parser.add_argument('--ds', '--datasets', nargs='+', default='2015.conll.freebase.horus 2016.conll.freebase.ascii.txt.horus ner.txt.horus emerging.test.annotated.horus', help='the horus datasets files: e.g.: ritter.horus wnut15.horus')
    #parser.add_argument('--ds', '--datasets', nargs='+', default='test.horus')
    parser.add_argument('--ds', '--datasets', nargs='+', default='2015.conll.freebase.horus')
    parser.add_argument('--exp', '--experiment_folder', default='EXP_005', action='store_true', required=False, help='the sub-folder name where the input file is located')
    parser.add_argument('--dt',       '--rundt',       action='store_true', required=False, default=1, help='benchmarks DT')
    parser.add_argument('--crf',      '--runcrf',      action='store_true', required=False, default=0, help='benchmarks CRF')
    parser.add_argument('--lstm',     '--runlstm',     action='store_true', required=False, default=0, help='benchmarks LSTM')
    parser.add_argument('--stanford', '--runstanford', action='store_true', required=False, default=0, help='benchmarks Stanford NER')

    parser.print_help()
    args = parser.parse_args()
    time.sleep(1)


    try:

        RUN_PROCESS_KEY_STARTS = 1
        RUN_PROCESS_KEY_ENDS = max(dict_exp_configurations.keys())
        # RUN_PROCESS_KEY_ENDS = 14

        k = range(RUN_PROCESS_KEY_STARTS, RUN_PROCESS_KEY_ENDS+1)
        k = [41] # best standard + horus 1.0 + horus 2.0

        #benchmark(experiment_folder=args.exp, runCRF=bool(args.crf), runRF=bool(args.dt),
        #          runLSTM=bool(args.lstm), runSTANFORD_NER=bool(args.stanford), optimization=True, keys=k)
        benchmark(experiment_folder=args.exp, runCRF=True, runRF=False,
                  runLSTM=False, runSTANFORD_NER=False, optimization=True, keys=k)
    except:
        raise

if __name__ == "__main__":
    #benchmark_no_train()

    main()
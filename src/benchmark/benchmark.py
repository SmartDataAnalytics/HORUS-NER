import argparse
import os

import time

import gc

import keras
import sklearn
from keras.utils import to_categorical
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder

from src.util.definitions import encoder_le1_name, dict_exp_configurations, SET_MASK, schemaindex2label

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

def benchmark(experiment_folder, datasets, runCRF = False, runDT = False, runLSTM = False, runSTANFORD_NER = False):

    config.logger.info('models: CRF=%s, DT=%s, LSTM=%s, Stanford=%s' % (str(runCRF), str(runDT), str(runLSTM), str(runSTANFORD_NER)))
    experiment_folder+='/'

    #sorted_labels = definitions.KLASSES.copy()
    #del sorted_labels[4]
    sorted_labels={'PER': 'PER', 'ORG': 'ORG', 'LOC': 'LOC'}
    r = [42, 19, 10] #3 folds cross validation
    # hyper-parameters
    _crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.088, c2=0.002, max_iterations=100, all_possible_transitions=True)
    _crf2 = sklearn_crfsuite.CRF(algorithm='pa', all_possible_transitions=True)
    _dt = ensemble.RandomForestClassifier(n_estimators=50)
    # 0 should be different to idx of 'O' IMO! so I keep 0 for data padding.
    temp = [0]
    temp.extend(definitions.PLOMNone_index2label.keys())
    one_hot_encode_y = to_categorical(temp)
    embedding_size = 128
    hidden_size = 32
    batch_size = 128
    enc_word = joblib.load(config.dir_encoders + definitions.encoder_int_words_name)
    epochs = 50
    verbose = 0
    #_meta = MEX('HORUS_EMNLP', _label, 'meta and multi-level machine learning for NLP')
    RUN_PROCESS_KEY_STARTS = 1
    RUN_PROCESS_KEY_ENDS = max(dict_exp_configurations.keys())
    #RUN_PROCESS_KEY_ENDS = 30
    header = 'cross-validation\tconfig\trun\tlabel\tprecision\trecall\tf1\tsupport\talgo\tdataset1\tdataset2\ttask\n'
    line = '%s\t%s\t%s\t%s\t%.5f\t%.5f\t%.5f\t%s\t%s\t%s\t%s\t%s\n'


    config.logger.info('running benchmark...')
    # benchmark starts
    name='metadata_'
    if runCRF: name+='crf_'
    if runDT: name+='dt_'
    if runLSTM: name+='lstm_'
    assert name != 'metadata_'

    name += ''.join(map(str,(range(RUN_PROCESS_KEY_STARTS, RUN_PROCESS_KEY_ENDS+1))))
    name +='.txt'
    out_file = open(config.dir_output + name, 'w+')
    out_file.write(header)
    for f_key in range(RUN_PROCESS_KEY_STARTS, RUN_PROCESS_KEY_ENDS+1):
        config.logger.info('loading dumps for configuration: ' + str(f_key))
        try:
            for ds1 in datasets:
                horus_m4_name_ds1 = ds1[0]
                horus_m4_path_ds1 = ds1[1].replace('.horusx', '.horus4') + ds1[2]

                dump_name = SET_MASK % (horus_m4_name_ds1, str(f_key))

                dump_full_path_ds1_sentence = os.path.dirname(os.path.realpath(horus_m4_path_ds1)) + '/' + \
                                              dump_name.replace('.pkl', '.sentence.pkl')
                dump_full_path_ds1_sentence_idx = os.path.dirname(os.path.realpath(horus_m4_path_ds1)) + '/' + \
                                              dump_name.replace('.pkl', '.sentence.enc.pkl')
                dump_full_path_ds1_token = os.path.dirname(os.path.realpath(horus_m4_path_ds1)) + '/' + \
                                           dump_name.replace('.pkl', '.token.pkl')
                dump_full_path_ds1_crf = os.path.dirname(os.path.realpath(horus_m4_path_ds1)) + '/' + \
                                         dump_name.replace('.pkl', '.crf.pkl')

                if not os.path.isfile(dump_full_path_ds1_sentence):
                    config.logger.info(dump_full_path_ds1_sentence)
                    config.logger.error(' -- configuration file does not exist! check its creation')
                    raise Exception
                else:
                    config.logger.debug('loading: ' + dump_name + ' dump files')

                    with open(dump_full_path_ds1_sentence, 'rb') as input:
                        file_name, f_key, X1_sentence, Y1_sentence = pickle.load(input)
                        # just PLO
                        temp = [['O' if item == 'MISC' else item for item in lst] for lst in Y1_sentence]
                        Y1_sentence = temp

                    with open(dump_full_path_ds1_sentence_idx, 'rb') as input:
                        file_name, f_key, X1_sentence_idx, Y1_sentence_idx = pickle.load(input)
                        # just PLO
                        #temp = [[definitions.PLOMNone_label2index['O'] if item == definitions.PLOMNone_label2index['MISC']
                        #         else item for item in lst] for lst in Y1_sentence_idx]
                        #Y1_sentence_idx = temp

                    with open(dump_full_path_ds1_token, 'rb') as input:
                        file_name, f_key, X1_token, Y1_token = pickle.load(input)
                        # just PLO
                        temp = [definitions.PLOMNone_label2index['O'] if y==definitions.PLOMNone_label2index['MISC']
                                else y for y in Y1_token]
                        Y1_token = temp

                    with open(dump_full_path_ds1_crf, 'rb') as input:
                        file_name, f_key, X1_crf, Y1_crf = pickle.load(input)
                        Y1_crf = Y1_sentence


                    #_set_name = SET_MASK % (ds1, str(f_key))
                    ##_file = config.dir_output + experiment_folder + _set_name
                    ##config.logger.info('ds1: loading [%s]: %s' % (ds1_name, _file))
                    ##with open(_file, 'rb') as input:
                    #    #shaped = pickle.load(input)
                    #ds1_config_name = dump_configs[_set_name][0]
                    #ds1_key = dump_configs[_set_name][1]
                    #X1_sentence = dump_configs[_set_name][2]
                    #Y1_sentence = dump_configs[_set_name][3]
                    #X1_token = dump_configs[_set_name][4]
                    #Y1_token = dump_configs[_set_name][5]
                    #X1_crf = dump_configs[_set_name][6]
                    #_Y1_sentence = dump_configs[_set_name][7]
                    #pca = PCA(n_components=50)
                    #X1_token_PCA = pca.fit(X1_token)
                    for ds2 in datasets:
                        horus_m4_name_ds2 = ds2[0]
                        horus_m4_path_ds2 = ds2[1].replace('.horusx', '.horus4') + ds2[2]
                        dump_name = SET_MASK % (horus_m4_name_ds2, str(f_key))

                        dump_full_path_ds2_sentence = os.path.dirname(
                            os.path.realpath(horus_m4_path_ds2)) + '/' + dump_name.replace('.pkl', '.sentence.pkl')
                        dump_full_path_ds2_sentence_idx = os.path.dirname(
                            os.path.realpath(horus_m4_path_ds2)) + '/' + dump_name.replace('.pkl', '.sentence.enc.pkl')
                        dump_full_path_ds2_token = os.path.dirname(
                            os.path.realpath(horus_m4_path_ds2)) + '/' + dump_name.replace('.pkl', '.token.pkl')
                        dump_full_path_ds2_crf = os.path.dirname(
                            os.path.realpath(horus_m4_path_ds2)) + '/' + dump_name.replace('.pkl', '.crf.pkl')


                        if not os.path.isfile(dump_full_path_ds2_sentence):
                            config.logger.info(dump_full_path_ds2_sentence)
                            config.logger.error(' -- configuration file does not exist! check its creation')
                            raise Exception
                        else:
                            config.logger.info('%s -> %s' % (horus_m4_name_ds1, horus_m4_name_ds2))
                            if horus_m4_name_ds1 != horus_m4_name_ds2:
                                config.logger.debug('loading: ' + dump_name + ' dump files')

                                with open(dump_full_path_ds2_sentence, 'rb') as input:
                                    file_name, f_key, X2_sentence, Y2_sentence = pickle.load(input)
                                    # just PLO
                                    temp = [['O' if item == 'MISC' else item for item in lst] for lst in Y2_sentence]
                                    Y2_sentence = temp

                                with open(dump_full_path_ds2_sentence_idx, 'rb') as input:
                                    file_name, f_key, X2_sentence_idx, Y2_sentence_idx = pickle.load(input)
                                    # just PLO
                                    #temp = [[definitions.PLOMNone_label2index['O'] if item ==
                                    #                                                  definitions.PLOMNone_label2index[
                                    #                                                      'MISC']
                                    #         else item for item in lst] for lst in Y2_sentence_idx]
                                    #Y2_sentence_idx = temp

                                with open(dump_full_path_ds2_token, 'rb') as input:
                                    file_name, f_key, X2_token, Y2_token = pickle.load(input)
                                    # just PLO
                                    temp = [
                                        definitions.PLOMNone_label2index['O'] if y == definitions.PLOMNone_label2index[
                                            'MISC']
                                        else y for y in Y2_token]
                                    Y2_token = temp

                                with open(dump_full_path_ds2_crf, 'rb') as input:
                                    file_name, f_key, X2_crf, Y2_crf = pickle.load(input)
                                    Y2_crf = Y2_sentence


                            if horus_m4_name_ds1 != horus_m4_name_ds2:
                                #_set_name = SET_MASK % (ds2_name, str(f_key))
                                ##_file = config.dir_output + experiment_folder + _set_name
                                ##config.logger.info('ds2: loading [%s]: %s' % (ds2_name, _file))
                                ##with open(_file, 'rb') as input:
                                ##    shaped = pickle.load(input)
                                #ds2_config_name = dump_configs[_set_name][0]
                                #ds2_key = dump_configs[_set_name][1]
                                #X2_sentence = dump_configs[_set_name][2]
                                #Y2_sentence = dump_configs[_set_name][3]
                                #X2_token = dump_configs[_set_name][4]
                                #Y2_token = dump_configs[_set_name][5]
                                #X2_crf = dump_configs[_set_name][6]
                                #_Y2_sentence = dump_configs[_set_name][7]

                                if (X2_token is None or X2_token.empty or X2_token.empty) is True:
                                    raise Exception('X_token error!')

                                # ---------------------------------------------------------- META ----------------------------------------------------------
                                # _conf = MEXConfiguration(id=len(_meta.configurations) + 1, horus_enabled=int(horus_feat),
                                #                         dataset_train=ds1[0], dataset_test=ds2[0] ,features=ds1[1], cross_validation=0)
                                # --------------------------------------------------------------------------------------------------------------------------
                                if runDT is True:
                                    m = _dt.fit(X1_token, Y1_token)
                                    ypr = m.predict(X2_token)
                                    # print(skmetrics.classification_report(Y2_token , ypr, labels=PLO_KLASSES.keys(), target_names=PLO_KLASSES.values(), digits=3))
                                    P, R, F, S = \
                                        sklearn.metrics.precision_recall_fscore_support(Y2_token, np.array(ypr).astype(int),
                                                                                        labels=definitions.PLOM_index2label.keys())
                                    for k in range(len(P)):
                                        out_file.write(line % ('False', str(f_key), '1', definitions.PLOM_index2label.get(k + 1),
                                                               P[k], R[k], F[k], str(S[k]), 'DT', horus_m4_name_ds1, horus_m4_name_ds2, 'NER'))

                                    # average
                                    P_avg, R_avg, F_avg, S_avg = sklearn.metrics.precision_recall_fscore_support(Y2_token, np.array(ypr).astype(int),
                                                                                                                 labels=definitions.PLOM_index2label.keys(), average='weighted')
                                    out_file.write(line % (
                                    'False', str(f_key), '1', 'average', P_avg, R_avg, F_avg, 0, 'DT', horus_m4_name_ds1, horus_m4_name_ds2, 'NER'))

                                    # entity detection only
                                    ypr_bin = [1 if x in definitions.PLOM_index2label.keys() else 0 for x in ypr]
                                    y2_bin = [1 if x in definitions.PLOM_index2label.keys() else 0 for x in Y2_token]
                                    P, R, F, S = sklearn.metrics.precision_recall_fscore_support(y2_bin, ypr_bin)
                                    for k in range(len(P)):
                                        out_file.write(line % ('False', str(f_key), '1', k,
                                                               P[k], R[k], F[k], str(S[k]), 'DT', horus_m4_name_ds1, horus_m4_name_ds2, 'NED'))

                                    # ---------------------------------------------------------- META ----------------------------------------------------------
                                    # _ex = MEXExecution(id=len(_conf.executions) + 1, alg='DT', phase='test', random_state=r[d])
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
                                        out_file.write(line % (
                                        'False', str(f_key), '1', definitions.PLOM_index2label.get(k + 1), P[k], R[k], F[k], str(S[k]),
                                        'CRF', horus_m4_name_ds1, horus_m4_name_ds2, 'NER'))

                                    # average
                                    P_avg, R_avg, F_avg, S_avg = sklearn.metrics.precision_recall_fscore_support(
                                        _yte, _ypr,
                                        labels=definitions.PLOM_index2label.values(), average='weighted')
                                    out_file.write(line % (
                                        'False', str(f_key), '1', 'average', P_avg, R_avg, F_avg, 0, 'CRF',
                                        horus_m4_name_ds1, horus_m4_name_ds2, 'NER'))

                                    # entity detection only
                                    ypr_bin = [1 if x in definitions.PLOM_index2label.values() else 0 for x in _ypr]
                                    y2_bin = [1 if x in definitions.PLOM_index2label.values() else 0 for x in _yte]
                                    P, R, F, S = sklearn.metrics.precision_recall_fscore_support(y2_bin, ypr_bin)
                                    for k in range(len(P)):
                                        out_file.write(line % (
                                            'False', str(f_key), '1', k, P[k], R[k], F[k], str(S[k]), 'CRF', horus_m4_name_ds1, horus_m4_name_ds2, 'NED'))


                                    m = _crf2.fit(X1_crf, Y1_sentence)
                                    ypr = m.predict(X2_crf)
                                    _ypr = np.array([tag for row in ypr for tag in row])
                                    _yte = np.array([tag for row in Y2_sentence for tag in row])
                                    P, R, F, S = sklearn.metrics.precision_recall_fscore_support(_yte, _ypr,
                                                                                                 labels=definitions.PLOM_index2label.values())
                                    for k in range(len(P)):
                                        out_file.write(line % (
                                        'False', str(f_key), '1', definitions.PLOM_index2label.get(k + 1), P[k], R[k], F[k], str(S[k]),
                                        'CRF_PA', horus_m4_name_ds1, horus_m4_name_ds2, 'NER'))

                                    # entity detection only
                                    ypr_bin = [1 if x in definitions.PLOM_index2label.values() else 0 for x in _ypr]

                                    P, R, F, S = sklearn.metrics.precision_recall_fscore_support(y2_bin, ypr_bin)
                                    for k in range(len(P)):
                                        out_file.write(line % (
                                            'False', str(f_key), '1', k, P[k], R[k], F[k], str(S[k]), 'CRF_PA', horus_m4_name_ds1, horus_m4_name_ds2, 'NED'))

                                if runLSTM is True:
                                    print(1)
                                    #max_of_sentences = max(maxlen_1, maxlen_2)
                                    #X2_lstm = pad_sequences(X2_lstm, maxlen=max_of_sentences)
                                    #y2_lstm = pad_sequences(y2_lstm, maxlen=max_of_sentences)
                                    #run_lstm(X1_lstm, X2_lstm, y1_lstm, y2_lstm, max_features_1, max_features_2, out_size_1,
                                    #         embedding_size, hidden_size, batch_size, epochs, verbose, max_of_sentences)
                            else:
                                # same dataset (cross-validation)

                                # ---------------------------------------------------------- META ----------------------------------------------------------
                                # _conf = MEXConfiguration(id=len(_meta.configurations)+1, horus_enabled=int(horus_feat),
                                #                                dataset_train=ds1[0], dataset_test=ds1[0], dataset_validation=None, features=None, cross_validation=1)
                                # --------------------------------------------------------------------------------------------------------------------------
                                if X1_token.empty is True:
                                    raise Exception('X1_token is empty!')

                                for d in range(len(r)):
                                    if runDT is True:
                                        Xtr, Xte, ytr, yte = train_test_split(X1_token, Y1_token, test_size=ds_test_size,
                                                                              random_state=r[d])
                                        m = _dt.fit(np.array(Xtr).astype(float), np.array(ytr).astype(int))
                                        # print(m.feature_importances_)
                                        ypr = m.predict(np.array(Xte).astype(float))
                                        # print(skmetrics.classification_report(np.array(yte).astype(int), np.array(ypr).astype(int), labels=definitions.PLO_KLASSES.keys(), target_names=definitions.PLO_KLASSES.values(), digits=3))
                                        P, R, F, S = sklearn.metrics.precision_recall_fscore_support(np.array(yte).astype(int),
                                                                                                     np.array(ypr).astype(int),
                                                                                                     labels=definitions.PLOM_index2label.keys())
                                        for k in range(len(P)):
                                            out_file.write(line % ('True', str(f_key), str(d + 1), definitions.PLOM_index2label.get(k + 1),
                                                        P[k], R[k], F[k], str(S[k]), 'DT', horus_m4_name_ds1, horus_m4_name_ds2, 'NER'))

                                        # average
                                        P_avg, R_avg, F_avg, S_avg = sklearn.metrics.precision_recall_fscore_support(
                                            np.array(yte).astype(int), np.array(ypr).astype(int),
                                            labels=definitions.PLOM_index2label.keys(),
                                            average='weighted')
                                        out_file.write(line % ('True', str(f_key), '0', 'average', P_avg, R_avg, F_avg, 0, 'DT', horus_m4_name_ds1, horus_m4_name_ds2, 'NER'))

                                        # entity detection only
                                        ypr_bin = [1 if x in definitions.PLOM_index2label.keys() else 0 for x in ypr]
                                        y2_bin = [1 if x in definitions.PLOM_index2label.keys() else 0 for x in yte]
                                        P, R, F, S = sklearn.metrics.precision_recall_fscore_support(y2_bin, ypr_bin)
                                        for k in range(len(P)):
                                            out_file.write(line % ('True', str(f_key), str(d + 1), k, P[k], R[k], F[k], str(S[k]),
                                                                   'DT', horus_m4_name_ds1, horus_m4_name_ds2, 'NED'))

                                        # ---------------------------------------------------------- META ----------------------------------------------------------
                                        # _ex = MEXExecution(id=len(_conf.executions) + 1, model='', alg='DT', phase='test', random_state=r[d])
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
                                        #print(metrics.flat_classification_report(yte, ypr, labels=sorted_labels.keys(), target_names=sorted_labels.values(), digits=3))
                                        P, R, F, S = sklearn.metrics.precision_recall_fscore_support(_yte, _ypr,
                                                                                                     labels=definitions.PLOM_index2label.values())
                                        for k in range(len(P)):
                                            out_file.write(line % (
                                                'True', str(f_key), str(d + 1), definitions.PLOM_index2label.get(k + 1), P[k], R[k],
                                                F[k],
                                                str(S[k]), 'CRF', horus_m4_name_ds1, horus_m4_name_ds2, 'NER'))

                                        # average
                                        P_avg, R_avg, F_avg, S_avg = sklearn.metrics.precision_recall_fscore_support(_yte, _ypr, labels=definitions.PLOM_index2label.values(), average='weighted')
                                        out_file.write(line % ('True', str(f_key), '0', 'average', P_avg, R_avg, F_avg, 0, 'CRF',horus_m4_name_ds1, horus_m4_name_ds2, 'NER'))

                                        # entity detection only
                                        ypr_bin = [1 if x in definitions.PLOM_index2label.values() else 0 for x in _ypr]
                                        y2_bin = [1 if x in definitions.PLOM_index2label.values() else 0 for x in _yte]
                                        P, R, F, S = sklearn.metrics.precision_recall_fscore_support(y2_bin, ypr_bin)
                                        for k in range(len(P)):
                                            out_file.write(line % (
                                                'True', str(f_key), str(d + 1), k, P[k], R[k], F[k], str(S[k]), 'CRF', horus_m4_name_ds1, horus_m4_name_ds2, 'NED'))

                                        m = _crf2.fit(Xtr, ytr)
                                        ypr = m.predict(Xte)
                                        _ypr = np.array([tag for row in ypr for tag in row])
                                        _yte = np.array([tag for row in yte for tag in row])
                                        P, R, F, S = sklearn.metrics.precision_recall_fscore_support(_yte, _ypr,
                                                                                                     labels=definitions.PLOM_index2label.values())
                                        for k in range(len(P)):
                                            out_file.write(line % (
                                                'True', str(f_key), str(d + 1), definitions.PLOM_index2label.get(k + 1), P[k], R[k],
                                                F[k],
                                                str(S[k]), 'CRF_PA', horus_m4_name_ds1, horus_m4_name_ds2, 'NER'))

                                        # average
                                        P_avg, R_avg, F_avg, S_avg = sklearn.metrics.precision_recall_fscore_support(
                                            _yte, _ypr, labels=definitions.PLOM_index2label.values(),
                                            average='weighted')
                                        out_file.write(line % (
                                        'True', str(f_key), '0', 'average', P_avg, R_avg, F_avg, 0, 'CRF_PA',
                                        horus_m4_name_ds1, horus_m4_name_ds2, 'NER'))

                                        # entity detection only
                                        ypr_bin = [1 if x in definitions.PLOM_index2label.values() else 0 for x in _ypr]
                                        y2_bin = [1 if x in definitions.PLOM_index2label.values() else 0 for x in _yte]
                                        P, R, F, S = sklearn.metrics.precision_recall_fscore_support(y2_bin, ypr_bin)
                                        for k in range(len(P)):
                                            out_file.write(line % (
                                                'True', str(f_key), str(d + 1), k, P[k], R[k], F[k], str(S[k]), 'CRF_PA', horus_m4_name_ds1, horus_m4_name_ds2, 'NED'))

                                        # ---------------------------------------------------------- META ----------------------------------------------------------
                                        # _ex = MEXExecution(id=len(_conf.executions)+1, model='', alg='CRF', phase='test', random_state=r[d])
                                        # P, R, F, S = sklearn.metrics.precision_recall_fscore_support(yte, ypr, labels=sorted_labels.keys(), average=None)
                                        # for k in sorted_labels.keys():
                                        #    _ex.add_performance(MEXPerformance(k, P[k], R[k], F[k], 0.0, S[k]))
                                        # _conf.add_execution(_ex)
                                        # _meta.add_configuration(_conf)
                                        # --------------------------------------------------------------------------------------------------------------------------

                                    if runLSTM is True:
                                        n_features = len(X1_sentence_idx[0].columns)
                                        lengths = [len(x) for x in X1_sentence_idx]
                                        max_len = max(lengths)
                                        n_words = len(enc_word.classes_)
                                        out_size = len(definitions.PLOMNone_label2index) + 1

                                        # idx word = 85 (word.lemma)
                                        seq_words = [S[85].tolist() for S in X1_sentence]
                                        import itertools
                                        flatt = list(itertools.chain(*seq_words))
                                        flatt = list(set(flatt))
                                        flatt.append("ENDPAD")

                                        len_words = len(flatt)
                                        word2idx = {w: i + 1 for i, w in enumerate(flatt)}
                                        X1_sentence_words = [[word2idx[w] for w in s] for s in seq_words]
                                        X1_sentence_words_pad = pad_sequences(sequences=X1_sentence_words,
                                                                              maxlen=max_len, padding='post',
                                                                              value=len_words-1)

                                        #horus features


                                        X1_sentence_idx_pad = pad_sequences(sequences=[x.values.tolist() for x in X1_sentence_idx],
                                                                            maxlen=max_len, padding='post',
                                                                            value=0)
                                        y1_idx = [[definitions.PLOMNone_label2index[y] for y in s] for s in Y1_sentence]
                                        y1_idx_pad = pad_sequences(sequences=y1_idx, maxlen=max_len, padding='post',
                                                                            value=0) #value=definitions.PLOMNone_label2index['O']

                                        y1_idx_pad_enc = [one_hot_encode_y[y] for y in y1_idx_pad]
                                        Xtr, Xte, ytr, yte = train_test_split(X1_sentence_idx_pad, y1_idx_pad_enc,
                                                                              test_size=ds_test_size, random_state=r[d])

                                        # define the BiLSTM+CRF architecture

                                        #input1 = Input(shape=(max_len,))
                                        #word_emb = Embedding(input_dim=n_words + 1, output_dim=150,
                                        #                     input_length=max_len,
                                        #                     mask_zero=True)(input1)

                                        input2 = Input(shape=(max_len,))
                                        horus_emb = Embedding(input_dim=100000, output_dim=20,
                                                              input_length=max_len,
                                                              mask_zero=True)(input2)

                                        #concat_layer = keras.layers.concatenate([word_emb, horus_emb], axis=1)

                                        model = Bidirectional(LSTM(units=50, return_sequences=True,
                                                                   recurrent_dropout=0.1))(horus_emb)
                                        model = Dense(150, activation='relu')(model)
                                        model = Dropout(0.1, noise_shape=None, seed=None)(model)
                                        model = Dense(150, activation='relu')(model)
                                        #model = TimeDistributed(Dense(50, activation='relu'))(model)
                                        crf = CRF(out_size)
                                        out = crf(model)

                                        #model = Model(inputs=[input1, input2], outputs=[out])
                                        model = Model(inputs=input2, outputs=out)
                                        model.compile(optimizer='rmsprop', loss=crf.loss_function,
                                                      metrics=[crf.accuracy])
                                        print(model.summary())

                                        hist = model.fit(Xtr[:, :, 0], np.array(ytr), batch_size=64, epochs=50,
                                                         validation_split=0.2, verbose=0)
                                        from seqeval.metrics import precision_score, recall_score, f1_score, \
                                            classification_report
                                        test_pred = model.predict(Xte[:, :, 0], verbose=1)

                                        pred_labels = pred2label(test_pred)
                                        test_labels = pred2label(yte)
                                        #fone = f1_score(test_labels, pred_labels)
                                        #print(classification_report(test_labels, pred_labels, digits=3))

                                        _ypr_nn = np.array([tag for row in pred_labels for tag in row])
                                        _yte_nn = np.array([tag for row in test_labels for tag in row])
                                        P, R, F, S = sklearn.metrics.precision_recall_fscore_support(_yte_nn, _ypr_nn,
                                                                        labels=definitions.PLOM_index2label.values())

                                        print(sklearn.metrics.classification_report(_yte_nn, _ypr_nn,
                                                                                 labels=definitions.PLOM_index2label.values(),
                                                                                 digits=3))

                                        for k in range(len(P)):
                                            out_file.write(line % (
                                                'True', str(f_key), str(d + 1), definitions.PLOM_index2label.get(k + 1),
                                                P[k], R[k],
                                                F[k],
                                                str(S[k]), 'BiLSTM+CRF', horus_m4_name_ds1, horus_m4_name_ds2, 'NER'))

                                        # average
                                        P_avg, R_avg, F_avg, S_avg = sklearn.metrics.precision_recall_fscore_support(
                                            _yte_nn, _ypr_nn, labels=definitions.PLOM_index2label.values(),
                                            average='weighted')
                                        out_file.write(line % (
                                        'True', str(f_key), '0', 'average', P_avg, R_avg, F_avg, 0, 'BiLSTM+CRF',
                                        horus_m4_name_ds1, horus_m4_name_ds2, 'NER'))

                                        '''
                                        m1 = Sequential()
                                        m1.add(Embedding(input_dim=n_words + 1, output_dim=20,
                                                          input_length=max_len,
                                                          mask_zero=True))

                                        m2 = Sequential()
                                        m2.add(InputLayer(input_shape=(max_len, 1)))


                                        merged = Concatenate([m1, m2])
                                        model = Model(merged, )

                                        model.add(Bidirectional(LSTM(units=50, return_sequences=True,
                                                                   recurrent_dropout=0.1,
                                                                   input_shape=(max_len, n_features))))
                                        model.add(TimeDistributed(Dense(50, activation='relu')))
                                        model.add(CRF(out_size))
                                        model.compile(optimizer='rmsprop', loss=crf.loss_function,
                                                      metrics=[crf.accuracy])
                                        print(model.summary())




                                        #    keras.layers.concatenate([m1, m2])


                                        input = Input(shape=(max_len, 1))
                                        word_emb = Embedding(input_dim=n_words + 1, output_dim=20,
                                                          input_length=max_len,
                                                          mask_zero=True)(input)

                                        feat_input = Input(shape=(max_len, n_features))
                                        feat_emb = Embedding(input_dim=1000, output_dim=20, input_length=max_len,
                                                             mask_zero=True)(feat_input)

                                        model = keras.layers.concatenate([word_emb, feat_emb])

                                        model = Bidirectional(LSTM(units=50, return_sequences=True,
                                                                   recurrent_dropout=0.1,
                                                                   input_shape=(max_len, n_features)))(model)
                                        model = TimeDistributed(Dense(50, activation='relu'))(model)
                                        crf = CRF(out_size)

                                        out = crf(model)

                                        model = Model(input, out)
                                        model.compile(optimizer='rmsprop', loss=crf.loss_function, metrics=[crf.accuracy])
                                        print(model.summary())

                                        hist = model.fit(Xtr, np.array(ytr), batch_size=32, epochs=5, validation_split=0.1, verbose=1)

                                        #y1_enc, y2_enc = encode_labels(max_features, ytr, yte)

                                        run_lstm(Xtr, Xte, np.array(ytr), np.array(yte),
                                                 n_words, n_words, out_size, embedding_size,
                                                 hidden_size, batch_size, epochs, verbose, max_len)
                                        '''

                        out_file.flush()
        except Exception as e:
            config.logger.error(repr(e))
    out_file.close()
    #with open(_label + '.meta', 'wb') as handle:
    #    pickle.dump(_meta, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
    parser.add_argument('--dt',       '--rundt',       action='store_true', required=False, default=0, help='benchmarks DT')
    parser.add_argument('--crf',      '--runcrf',      action='store_true', required=False, default=0, help='benchmarks CRF')
    parser.add_argument('--lstm',     '--runlstm',     action='store_true', required=False, default=1, help='benchmarks LSTM')
    parser.add_argument('--stanford', '--runstanford', action='store_true', required=False, default=0, help='benchmarks Stanford NER')

    parser.print_help()
    args = parser.parse_args()
    time.sleep(1)



    try:

        benchmark(experiment_folder=args.exp, datasets=definitions.NER_DATASETS, runCRF=bool(args.crf), runDT=bool(args.dt),
                  runLSTM=bool(args.lstm), runSTANFORD_NER=bool(args.stanford))
    except:
        raise

if __name__ == "__main__":
    main()
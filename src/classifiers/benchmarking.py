import argparse
import os

import time

import sklearn
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer

from src.classifiers.experiment_metadata import MEXExecution, \
    MEXPerformance, MEX, MEXConfiguration

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import matplotlib.pyplot as plt
import sklearn_crfsuite
from sklearn import ensemble
from sklearn import metrics as skmetrics
from sklearn.cross_validation import train_test_split, KFold
from sklearn_crfsuite import metrics

plt.style.use('ggplot')
from src.config import HorusConfig
from src.core.util import definitions
from sklearn.externals import joblib
import scipy.stats
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.grid_search import RandomizedSearchCV
import numpy as np
from keras.engine import InputLayer
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Merge
from nltk.corpus import stopwords
from nltk import LancasterStemmer, re
import pandas as pd
import pickle
import multiprocessing
from functools import partial
from contextlib import contextmanager
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
lancaster_stemmer = LancasterStemmer()
stop = set(stopwords.words('english'))
le1 = joblib.load(config.dir_encoders + "_encoder_pos.pkl")
le2 = joblib.load(config.dir_encoders + "_encoder_nltk2.pkl")

def to_check():
    # feature_extraction

    crf2 = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.18687907015736968,
        c2=0.025503200544851036,
        max_iterations=100,
        all_possible_transitions=True
    )
    # crf2.fit(X_train_CRF_shape, y_train_CRF_shape)

    # eval

    # y_pred2 = crf2.predict(X_test_CRF_shape)

    # metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)

    # labels = list(crf.classes_)
    # trick for report visualization

    # labels = list(['LOC', 'ORG', 'PER'])
    # labels.remove('O')
    # group B and I results
    # sorted_labels = sorted(
    #    labels,
    #    key=lambda name: (name[1:], name[0])
    # )



    # print(metrics.flat_classification_report(
    #    y_test, y_pred2, labels=sorted_labels, digits=3
    # ))
    exit(0)
    # r = [42, 39, 10, 5, 50]
    # fmeasures = []
    # for d in range(len(r)):
    #    cv_X_train, cv_X_test, cv_y_train, cv_y_test = train_test_split(X_train_CRF_shape, y_train_CRF_shape,
    #                                                        test_size = 0.30, random_state = r[d])
    #    m = crf.fit(cv_X_train, cv_y_train)
    #    cv_y_pred = m.predict(cv_X_test)
    #    print(metrics.flat_classification_report(
    #        cv_y_test, cv_y_pred, labels=sorted_labels, digits=3
    #    ))
    # cv_y_test_bin = MultiLabelBinarizer().fit_transform(cv_y_test)
    # cv_y_pred_bin = MultiLabelBinarizer().fit_transform(cv_y_pred)
    # fmeasures.append(f1_score(cv_y_test_bin, cv_y_pred_bin, average='weighted'))

    # print sum(fmeasures)/len(r)

    # scores = cross_val_score(crf, _X, _y, cv=5, scoring='f1_macro')
    # scores2 = cross_val_score(crf2, _X, _y, cv=5, scoring='f1_macro')

    # rs = ShuffleSplit(n_splits=3, test_size=.20, random_state=0)
    # for train_index, test_index in rs.split(_X):
    #    print("TRAIN:", train_index, "TEST:", test_index)

    # print scores
    # print scores2

    # exit(0)

    # define fixed parameters and parameters to search
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=300,
        all_possible_transitions=True
    )
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
    }

    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score,
                            average='weighted', labels=labels)

    # search
    rs = RandomizedSearchCV(crf, params_space,
                            cv=3,
                            verbose=1,
                            n_jobs=-1,
                            n_iter=50,
                            scoring=f1_scorer)
    rs.fit(X_train_CRF_shape, y_train_CRF_shape)

    # crf = rs.best_estimator_
    print('best params:', rs.best_params_)
    print('best CV score:', rs.best_score_)
    print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

    _x = [s.parameters['c1'] for s in rs.grid_scores_]
    _y = [s.parameters['c2'] for s in rs.grid_scores_]
    _c = [s.mean_validation_score for s in rs.grid_scores_]

    fig = plt.figure()
    fig.set_size_inches(12, 12)
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('C1')
    ax.set_ylabel('C2')
    ax.set_title("Randomized Hyperparameter Search CV Results (min={:0.3}, max={:0.3})".format(
        min(_c), max(_c)
    ))

    ax.scatter(_x, _y, c=_c, s=60, alpha=0.9, edgecolors=[0, 0, 0])
    fig.savefig('crf_optimization.png')

    print("Dark blue => {:0.4}, dark red => {:0.4}".format(min(_c), max(_c)))

    crf = rs.best_estimator_
    y_pred = crf.predict(X_test_CRF_shape)
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=labels, digits=3
    ))

    from collections import Counter

    def print_transitions(trans_features):
        for (label_from, label_to), weight in trans_features:
            print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

    print("Top likely transitions:")
    print_transitions(Counter(crf.transition_features_).most_common(20))

    print("\nTop unlikely transitions:")
    print_transitions(Counter(crf.transition_features_).most_common()[-20:])

    def print_state_features(state_features):
        for (attr, label), weight in state_features:
            print("%0.6f %-8s %s" % (weight, label, attr))

    print("Top positive:")
    print_state_features(Counter(crf.state_features_).most_common(30))

    print("\nTop negative:")
    print_state_features(Counter(crf.state_features_).most_common()[-30:])

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

def shape(word):
    word_shape = 0 #'other'
    if re.match('[0-9]+(\.[0-9]*)?|[0-9]*\.[0-9]+$', word):
        word_shape = 1 #'number'
    elif re.match('\W+$', word):
        word_shape = 2 #'punct'
    elif re.match('[A-Z][a-z]+$', word):
        word_shape = 3 #'capitalized'
    elif re.match('[A-Z]+$', word):
        word_shape = 4 # 'uppercase'
    elif re.match('[a-z]+$', word):
        word_shape = 5 #'lowercase'
    elif re.match('[A-Z][a-z]+[A-Z][a-z]+[A-Za-z]*$', word):
        word_shape = 6 #'camelcase'
    elif re.match('[A-Za-z]+$', word):
        word_shape = 7 #'mixedcase'
    elif re.match('__.+__$', word):
        word_shape = 8 # 'wildcard'
    elif re.match('[A-Za-z0-9]+\.$', word):
        word_shape = 9 # 'ending-dot'
    elif re.match('[A-Za-z0-9]+\.[A-Za-z0-9\.]+\.$', word):
        word_shape = 10 # 'abbreviation'
    elif re.match('[A-Za-z0-9]+\-[A-Za-z0-9\-]+.*$', word):
        word_shape = 11 #'contains-hyphen'

    return word_shape

def convert_lstm_shape(ds, y, horus_feat = False):
    config.logger.info('shaping to LSTM format...')
    if horus_feat == False:
        Xclean = [[[c[3], c[4], c[10], c[12], c[13], c[17], c[18], c[20], c[21]] for c in x] for x in ds]
    else:
        Xclean = [[[c[3], c[4], c[10], c[12], c[13], c[17], c[18], c[20], c[21], c[23], c[24], c[25], c[26], c[27], c[28], c[29], c[30], c[31], c[32]] for c in x] for x in ds]

    all_text = [c[0] for x in Xclean for c in x]
    all_text.extend([c[1] for x in Xclean for c in x])

    words = list(set(all_text))  # distinct tokens
    word2ind = {word: index for index, word in enumerate(words)}  # indexes of words
    ind2word = {index: word for index, word in enumerate(words)}
    #labels = list(set([c for x in y for c in x]))
    label2ind = definitions.KLASSES2
    ind2label = definitions.KLASSES
    print('Vocabulary size:', len(word2ind), len(label2ind))
    lengths = [len(x) for x in Xclean]
    maxlen = max(lengths)
    print('min sentence / max sentence: ', min(lengths), maxlen)
    if horus_feat == False:
        X_enc = [[[word2ind[c[0]], word2ind[c[1]], c[2], c[3], c[4], c[5], c[6], c[7], c[8]] for c in x] for x in Xclean]
    else:
        X_enc = [[[word2ind[c[0]], word2ind[c[1]], c[2], c[3], c[4], c[5], c[6], c[7], c[8],
                   c[9], c[10], c[11], c[12], c[13], c[14], c[15], c[16], c[17], c[18]] for c in x] for x in Xclean]

    max_label = max(label2ind.values()) + 1
    y_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y]
    y_enc = [[encode(c, max_label) for c in ey] for ey in y_enc]

    max_features = len(word2ind)
    out_size = len(label2ind) + 1

    return X_enc, y_enc, max_features, out_size, maxlen

def sent2label(sent):
    return [definitions.KLASSES[y] for y in sent]

def sent2features(sent):
    return [features_to_crf_shape(sent, i) for i in range(len(sent))]

def features_to_crf_shape(sent, i):

    features = {'bias': 1.0}
    features.update(dict(('f'+str(key), str(sent.iloc[i].at[key])) for key in np.sort(sent.columns.values)))

    if i > 0:
        features_pre = dict(('-1:f'+str(key), str(sent.iloc[i-1].at[key])) for key in np.sort(sent.columns.values))
        features.update(features_pre)
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        features_pos = dict(('+1:f'+str(key), str(sent.iloc[i+1].at[key])) for key in np.sort(sent.columns.values))
        features.update(features_pos)
    else:
        features['EOS'] = True

    return features

def shape_data((file, path, le)):
    '''
    shape the dataframe, adding further traditional features
    :param file: the horus features file
    :param path: the path
    :param le: the encoder
    :return: an updated dataset and a sentence-shaped dataset
    '''
    try:

        ds_sentences, y_sentences_shape = [], []
        _sent_temp_feat, _sent_temp_y = [], []
        #ds_tokens, y_tokens_shape = [], []

        fullpath=path+file
        config.logger.info('reading horus features file: ' + fullpath)
        df = pd.read_csv(fullpath, delimiter="\t", skiprows=1, header=None, keep_default_na=False, na_values=['_|_'])
        #print(len(df))
        df=df.drop(df[df[definitions.INDEX_IS_COMPOUND]==1].index)
        #print(len(df))
        oldsentid = df.iloc[0].at[definitions.INDEX_ID_SENTENCE]
        df = df.reset_index(drop=True)
        COLS = len(df.columns)

        df=pd.concat([df, pd.DataFrame(columns=range(COLS,(COLS+definitions.STANDARD_FEAT)))], axis=1)
        config.logger.info(len(df))
        for row in df.itertuples():
            index=row.Index
            if index % 500 == 0: config.logger.info(index)
            if df.loc[index, definitions.INDEX_ID_SENTENCE] != oldsentid:
                ds_sentences.append(_sent_temp_feat)
                _sent_temp_feat = []
                y_sentences_shape.append(_sent_temp_y)
                _sent_temp_y = []

            idsent = df.loc[index].at[definitions.INDEX_ID_SENTENCE]
            #token=df.loc[index, definitions.INDEX_TOKEN]

            if index > 1: prev_prev_serie = df.loc[index-2]
            if index > 0: prev_serie = df.loc[index-1]
            if index + 1 < len(df): next_serie = df.loc[index+1]
            if index + 2 < len(df): next_next_serie = df.loc[index+2]

            _t=[]
            # standard features
            if index > 0 and prev_serie.at[definitions.INDEX_ID_SENTENCE] == idsent:
                prev_pos = prev_serie.at[definitions.INDEX_POS]
                prev_pos_uni = prev_serie.at[definitions.INDEX_POS_UNI]
                prev_token = prev_serie.at[definitions.INDEX_TOKEN]
                prev_one_char_token = int(len(prev_token) == 1)
                prev_special_char = int(len(re.findall('(http://\S+|\S*[^\w\s]\S*)', prev_token)) > 0)
                prev_first_capitalized = int(prev_token[0].isupper())
                prev_capitalized = int(prev_token.isupper())
                prev_title = int(prev_token.istitle())
                prev_digit = int(prev_token.isdigit())
                prev_stop_words = int(prev_token in stop)
                prev_small = int(len(prev_token) <= 2)
                #prev_lemma = lancaster_stemmer.stem(prev_token.decode('utf-8', errors='replace'))
                prev_hyphen = int('-' in prev_token)
                prev_sh = shape(prev_token)
            else:
                prev_pos = ''
                prev_pos_uni = ''
                prev_token = ''
                prev_one_char_token = -1
                prev_special_char = -1
                prev_first_capitalized = -1
                prev_capitalized = -1
                prev_title = -1
                prev_digit = -1
                prev_stop_words = -1
                prev_small = -1
                #prev_lemma = -1
                prev_hyphen = -1
                prev_sh = -1

            if index > 1 and prev_prev_serie.at[definitions.INDEX_ID_SENTENCE] == idsent:
                prev_prev_pos = prev_prev_serie.at[definitions.INDEX_POS]
                prev_prev_pos_uni = prev_prev_serie.at[definitions.INDEX_POS_UNI]
                prev_prev_token = prev_prev_serie.at[definitions.INDEX_TOKEN]
                prev_prev_one_char_token = int(len(prev_prev_token) == 1)
                prev_prev_special_char = int(len(re.findall('(http://\S+|\S*[^\w\s]\S*)', prev_prev_token)) > 0)
                prev_prev_first_capitalized = int(prev_prev_token[0].isupper())
                prev_prev_capitalized = int(prev_prev_token.isupper())
                prev_prev_title = int(prev_prev_token.istitle())
                prev_prev_digit = int(prev_prev_token.isdigit())
                prev_prev_stop_words = int(prev_prev_token in stop)
                prev_prev_small = int(len(prev_prev_token) <= 2)
                #prev_prev_lemma = lancaster_stemmer.stem(prev_prev_token.decode('utf-8', errors='replace'))
                prev_prev_hyphen = int('-' in prev_prev_token)
                prev_prev_sh = shape(prev_prev_token)
            else:
                prev_prev_pos= ''
                prev_prev_pos_uni = ''
                #prev_prev_token= ''
                prev_prev_one_char_token= -1
                prev_prev_special_char= -1
                prev_prev_first_capitalized= -1
                prev_prev_capitalized= -1
                prev_prev_title= -1
                prev_prev_digit= -1
                prev_prev_stop_words= -1
                prev_prev_small= -1
                #prev_prev_lemma= -1
                prev_prev_hyphen= -1
                prev_prev_sh= -1

            if index + 1 < len(df) and next_serie.at[definitions.INDEX_ID_SENTENCE] == idsent:
                next_pos = next_serie.at[definitions.INDEX_POS]
                next_pos_uni = next_serie.at[definitions.INDEX_POS_UNI]
                next_token = next_serie.at[definitions.INDEX_TOKEN]
                next_one_char_token = int(len(next_token) == 1)
                next_special_char = int(len(re.findall('(http://\S+|\S*[^\w\s]\S*)', next_token)) > 0)
                next_first_capitalized = int(next_token[0].isupper())
                next_capitalized = int(next_token.isupper())
                next_title = int(next_token.istitle())
                next_digit = int(next_token.isdigit())
                next_stop_words = int(next_token in stop)
                next_small = int(len(next_token) <= 2)
                #next_lemma = lancaster_stemmer.stem(next_token.decode('utf-8'))
                next_hyphen = int('-' in next_token)
                next_sh = shape(next_token)
            else:
                next_pos = ''
                next_pos_uni = ''
                next_token = ''
                next_one_char_token = -1
                next_special_char = -1
                next_first_capitalized = -1
                next_capitalized = -1
                next_title = -1
                next_digit = -1
                next_stop_words = -1
                next_small = -1
                next_lemma = -1
                next_hyphen = -1
                next_sh = -1

            if index + 2 < len(df) and next_next_serie.at[definitions.INDEX_ID_SENTENCE] == idsent:
                next_next_pos = next_next_serie.at[definitions.INDEX_POS]
                next_next_pos_uni = next_next_serie.at[definitions.INDEX_POS_UNI]
                next_next_token = next_next_serie.at[definitions.INDEX_TOKEN]
                next_next_one_char_token = int(len(next_next_token) == 1)
                next_next_special_char = int(len(re.findall('(http://\S+|\S*[^\w\s]\S*)', next_next_token)) > 0)
                next_next_first_capitalized = int(next_next_token[0].isupper())
                next_next_capitalized = int(next_next_token.isupper())
                next_next_title = int(next_next_token.istitle())
                next_next_digit = int(next_next_token.isdigit())
                next_next_stop_words = int(next_next_token in stop)
                next_next_small = int(len(next_next_token) <= 2)
                #next_next_lemma = lancaster_stemmer.stem(next_next_token.decode('utf-8',errors='replace'))
                next_next_hyphen = int('-' in next_next_token)
                next_next_sh = shape(next_next_token)
            else:
                next_next_pos = ''
                next_next_pos_uni = ''
                next_next_token = ''
                next_next_one_char_token = -1
                next_next_special_char = -1
                next_next_first_capitalized = -1
                next_next_capitalized = -1
                next_next_title = -1
                next_next_digit = -1
                next_next_stop_words = -1
                next_next_small = -1
                next_next_lemma = -1
                next_next_hyphen = -1
                next_next_sh = -1

            # standard features -2+2 context
            _t.extend([le.transform(prev_pos), le.transform(prev_pos_uni), prev_one_char_token,
                     prev_special_char, prev_first_capitalized, prev_capitalized, prev_title, prev_digit,
                     prev_stop_words, prev_small, prev_hyphen, prev_sh,
                     le.transform(prev_prev_pos), le.transform(prev_prev_pos_uni),
                     prev_prev_one_char_token, prev_prev_special_char,
                     prev_prev_first_capitalized, prev_prev_capitalized, prev_prev_title, prev_prev_digit,
                     prev_prev_stop_words, prev_prev_small, prev_prev_hyphen, prev_prev_sh,
                     le.transform(next_pos), le.transform(next_pos_uni), next_one_char_token,
                     next_special_char, next_first_capitalized, next_capitalized, next_title, next_digit,
                     next_stop_words, next_small, next_hyphen, next_sh,
                     le.transform(next_next_pos), le.transform(next_next_pos_uni),
                     next_next_one_char_token, next_next_special_char, next_next_first_capitalized,
                     next_next_capitalized, next_next_title, next_next_digit, next_next_stop_words, next_next_small,
                     next_next_hyphen, next_next_sh])
            # standard
            _t.extend([le.transform(df.loc[index].at[definitions.INDEX_POS]), le.transform(df.loc[index].at[definitions.INDEX_POS_UNI]),
                      int(len(df.loc[index].at[definitions.INDEX_TOKEN]) == 1),
                      int(len(re.findall('(http://\S+|\S*[^\w\s]\S*)', df.loc[index].at[definitions.INDEX_TOKEN])) > 0),
                      int(str(df.loc[index].at[definitions.INDEX_TOKEN])[0].isupper()),
                      int(str(df.loc[index].at[definitions.INDEX_TOKEN]).isupper()),
                      int(str(df.loc[index].at[definitions.INDEX_TOKEN]).istitle()),
                      int(str(df.loc[index].at[definitions.INDEX_TOKEN]).isdigit()),
                      int(str(df.loc[index].at[definitions.INDEX_TOKEN]) in stop),
                      int(len(df.loc[index].at[definitions.INDEX_TOKEN]) <= 2),
                      int('-' in str(df.loc[index].at[definitions.INDEX_TOKEN])),
                      shape(str(df.loc[index].at[definitions.INDEX_TOKEN]))])

            #if len(f_indexes) !=0:
            #    _t.extend(df.loc[index][f_indexes])
            df.iloc[[index], COLS:(COLS + definitions.STANDARD_FEAT+1)] = _t

            #_t = pd.Series(_t, index=range(COLS, COLS+STANDARD_FEAT))
            #a=df.iloc[index]
            #a.update(_t)
            #df.iloc[index]=a

            #df.iloc[[index], COLS:(COLS+STANDARD_FEAT)+1] = _t

            #new=COLS
            #for i in range(len(_t)):
            #    df.loc[index, new] = _t[i]
            #    new += 1

            # NER class
            if df.loc[index].at[definitions.INDEX_TARGET_NER] == 'O': y=u'O'
            elif df.loc[index].at[definitions.INDEX_TARGET_NER] in definitions.NER_TAGS_LOC: y = u'LOC'
            elif df.loc[index].at[definitions.INDEX_TARGET_NER] in definitions.NER_TAGS_ORG: y = u'ORG'
            elif df.loc[index].at[definitions.INDEX_TARGET_NER] in definitions.NER_TAGS_PER: y = u'PER'
            else: y = u'O'

            oldsentid = idsent

            #ds_tokens.append(_t)
            #y_tokens_shape.append(definitions.KLASSES2[y])

            _sent_temp_feat.append(df.loc[index])
            _sent_temp_y.append(definitions.KLASSES2[y])

        # adding last sentence
        ds_sentences.append(_sent_temp_feat)
        y_sentences_shape.append(_sent_temp_y)

        y_tokens_shape = df[definitions.INDEX_TARGET_NER].copy()
        del df[definitions.INDEX_TARGET_NER]

        config.logger.info('total of sentences: ' + str(len(ds_sentences)))
        config.logger.info('total of tokens: ' + str(len(df)))
        return file, (ds_sentences, y_sentences_shape), (df, y_tokens_shape)
    except:
        raise

def shape_datasets(experiment_folder, datasets):
    ret = []
    job_args = []
    for ds in datasets:
        config.logger.info(ds)
        _file = config.dir_output + experiment_folder + '_' + ds + '_shaped.pkl'
        if os.path.isfile(_file):
            with open(_file, 'rb') as input:
                shaped = pickle.load(input)
                ret.append(shaped)
        else:
            job_args.append((ds, config.dir_output + experiment_folder, le1))
            #data=shape_data(ds, config.dir_output + experiment_folder, le1)

    if len(job_args) > 0:
        p = multiprocessing.Pool(8)
        asyncres = p.map(shape_data, job_args)
        config.logger.info(len(asyncres))
        for data in asyncres:
            _file = config.dir_output + experiment_folder + '_' + data[0] + '_shaped.pkl'
            with open(_file, 'wb') as output:
                pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)
            ret.append(data)

    config.logger.info(len(ret))
    return ret

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

    model.add(LSTM(hidden_size, return_sequences=True, input_shape=(maxsent, Xtr.shape[2] - 1)))
    model.add(TimeDistributed(Dense(out_size)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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

def exclude_columns(df, f_indexes):
    dfret=df.copy()
    for icol in dfret.columns:
        if icol not in f_indexes:
            dfret.drop(icol, axis=1, inplace=True)
    return dfret

def benchmark(experiment_folder, datasets, runCRF = False, runDT = False, runLSTM = False, runSTANFORD_NER = False):

    config.logger.info('models: CRF=%s, DT=%s, LSTM=%s, Stanford=%s' % (str(runCRF), str(runDT), str(runLSTM), str(runSTANFORD_NER)))
    experiment_folder+='/'
    out_file = open(config.dir_output + experiment_folder + 'metadata.txt', 'w+')
    config.logger.info('datasets: ' + str(datasets))
    datasets=datasets.split()
    #sorted_labels = definitions.KLASSES.copy()
    #del sorted_labels[4]
    sorted_labels={'PER': 'PER', 'ORG': 'ORG', 'LOC': 'LOC'}
    r = [42, 39, 10, 5, 50]
    # hyper-parameters
    _crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.088, c2=0.002, max_iterations=100, all_possible_transitions=True)
    _dt = ensemble.RandomForestClassifier(n_estimators=50)
    embedding_size = 128
    hidden_size = 32
    batch_size = 128
    epochs = 50
    verbose = 0

    _label='EXP_004'
    _meta = MEX('HORUS_EMNLP', _label, 'meta and multi-level machine learning for NLP')

    dict_exp_feat = {1: range(85,(85+definitions.STANDARD_FEAT)), 2: definitions.FEATURES_HORUS_BASIC_CV, 3: definitions.FEATURES_HORUS_BASIC_TX,
                    4: definitions.FEATURES_HORUS_CNN_CV, 5: definitions.FEATURES_HORUS_CNN_TX,
                    6: definitions.FEATURES_HORUS_EMB_TX + definitions.FEATURES_HORUS_STATS_TX,
                    7: definitions.FEATURES_HORUS_TX, 8: definitions.FEATURES_HORUS_CV,
                    9: definitions.FEATURES_HORUS_BASIC_AND_CNN, 10: definitions.FEATURES_HORUS}

    config.logger.info('shaping the datasets...')
    shaped_datasets = shape_datasets(experiment_folder, datasets) # ds_name, (X1, y1 [DT-shape]), (X2, y2 [CRF-shape]), (X3, y3 [NN-shape])
    config.logger.info('done! running experiment configurations')

    header = 'cross-validation\tconfig\trun\tlabel\tprecision\trecall\tf1\tsupport\talgo\tdataset1\tdataset2\n'
    line = '%s\t%s\t%s\t%s\t%.5f\t%.5f\t%.5f\t%s\t%s\t%s\t%s\n'

    out_file.write(header)

    for f_key, f_indexes in dict_exp_feat.iteritems():
        for ds1 in shaped_datasets:
            ds1_name = ds1[0]
            X1_sentence = ds1[1][0]
            Y1_sentence = [sent2label(s) for s in ds1[1][1]]
            X1_token = exclude_columns(ds1[2][0], f_indexes)
            X1_token.replace('O', 0, inplace=True)
            Y1_token = [definitions.KLASSES2[y] for y in ds1[2][1]]
            #X1_sentence.replace('O', 0, inplace=True)
            #pca = PCA(n_components=50)
            #X1_token_PCA = pca.fit(X1_token)
            if runCRF is True:
                X1_crf = [sent2features(exclude_columns(pd.DataFrame(s), f_indexes)) for s in X1_sentence]
            if runLSTM is True:
                X1_lstm, y1_lstm, max_features_1, out_size_1, maxlen_1 = convert_lstm_shape(X1_sentence, Y1_sentence, f_indexes)
            for ds2 in shaped_datasets:
                ds2_name = ds2[0]
                if ds1[0] != ds2[0]:
                    ds2_name = ds2[0]
                    X2_sentence = ds2[1][0]
                    Y2_sentence = [sent2label(s) for s in ds2[1][1]]
                    X2_token = exclude_columns(ds2[2][0], f_indexes)
                    X2_token.replace('O', 0, inplace=True)
                    Y2_token = [definitions.KLASSES2[y] for y in ds2[2][1]]

                    if runCRF is True:
                        #X2_crf = [sent2features(s, f_indexes) for s in ds2[1][0]]
                        X2_crf = [sent2features(exclude_columns(pd.DataFrame(s), f_indexes)) for s in X2_sentence]
                    if runLSTM is True:
                        X2_lstm, y2_lstm, max_features_2, out_size_2, maxlen_2 = convert_lstm_shape(ds2[1][0], ds2[1][1], f_indexes)
                        X1_lstm = pad_sequences(X1_lstm, maxlen=max(maxlen_1, maxlen_2))
                        y1_lstm = pad_sequences(y1_lstm, maxlen=max(maxlen_1, maxlen_2))

                    #else
                    # if runDT is True:
                    #    X2_dt = None
                    #    Y2_dt = None
                    #if runCRF is True: X2_crf = X1_crf
                    #if runLSTM is True:
                    #    X2_lstm, y2_lstm, max_features_2, out_size_2, maxlen_2 = convert_lstm_shape(ds2[1][0], ds2[1][1], horus_feat)
                    #    X1_lstm = pad_sequences(X1_lstm, maxlen=max(maxlen_1, maxlen_2))
                    #    y1_lstm = pad_sequences(y1_lstm, maxlen=max(maxlen_1, maxlen_2))


                '''
                ************************************************************************************************************************************** 
                run the models
                **************************************************************************************************************************************
                '''
                if ds1_name == ds2_name:
                    # ---------------------------------------------------------- META ----------------------------------------------------------
                    #_conf = MEXConfiguration(id=len(_meta.configurations)+1, horus_enabled=int(horus_feat),
                    #                                dataset_train=ds1[0], dataset_test=ds1[0], dataset_validation=None, features=None, cross_validation=1)
                    # --------------------------------------------------------------------------------------------------------------------------
                    for d in range(len(r)):
                        if runDT is True:
                            Xtr, Xte, ytr, yte = train_test_split(X1_token, Y1_token, test_size=ds_test_size, random_state=r[d])
                            m = _dt.fit(np.array(Xtr).astype(float), np.array(ytr).astype(int))
                            #print(m.feature_importances_)
                            ypr = m.predict(np.array(Xte).astype(float))
                            #print(skmetrics.classification_report(np.array(yte).astype(int), np.array(ypr).astype(int), labels=definitions.PLO_KLASSES.keys(), target_names=definitions.PLO_KLASSES.values(), digits=3))
                            P, R, F, S = sklearn.metrics.precision_recall_fscore_support(np.array(yte).astype(int),
                                                                                         np.array(ypr).astype(int),
                                                                                         labels=definitions.PLO_KLASSES.keys())
                            for k in range(len(P)):
                                out_file.write(line % ('True', str(f_key), str(d + 1), definitions.PLO_KLASSES.get(k + 1),
                                                       P[k], R[k], F[k], str(S[k]), 'DT', ds1_name, ds2_name))

                            # ---------------------------------------------------------- META ----------------------------------------------------------
                            #_ex = MEXExecution(id=len(_conf.executions) + 1, model='', alg='DT', phase='test', random_state=r[d])
                            #P, R, F, S = sklearn.metrics.precision_recall_fscore_support(yte, ypr,
                            #                                                             labels=sorted_labels.keys(),
                            #                                                             average=None)
                            #for k in sorted_labels.keys():
                            #    _ex.add_performance(MEXPerformance(k, P[k], R[k], F[k], 0.0, S[k]))
                            #_conf.add_execution(_ex)
                            #_meta.add_configuration(_conf)
                            # --------------------------------------------------------------------------------------------------------------------------

                        if runCRF is True:
                            from sklearn.preprocessing import MultiLabelBinarizer
                            _Y1_sentence = MultiLabelBinarizer().fit_transform(Y1_sentence)
                            Xtr, Xte, ytr, yte = train_test_split(X1_crf, Y1_sentence, test_size=ds_test_size, random_state=r[d])
                            m = _crf.fit(Xtr, ytr)
                            ypr = m.predict(Xte)

                            _ypr = np.array([tag for row in ypr for tag in row])
                            _yte = np.array([tag for row in yte for tag in row])

                            #print(metrics.flat_classification_report(yte, ypr, labels=sorted_labels.keys(), target_names=sorted_labels.values(), digits=3))

                            P, R, F, S = sklearn.metrics.precision_recall_fscore_support(_yte, _ypr, labels=definitions.PLO_KLASSES.keys())
                            for k in range(len(P)):
                                out_file.write(line % (
                                'True', str(f_key), str(d + 1), definitions.PLO_KLASSES.get(k + 1), P[k], R[k], F[k],
                                str(S[k]), 'CRF', ds1_name, ds2_name))

                            # ---------------------------------------------------------- META ----------------------------------------------------------
                            #_ex = MEXExecution(id=len(_conf.executions)+1, model='', alg='CRF', phase='test', random_state=r[d])
                            #P, R, F, S = sklearn.metrics.precision_recall_fscore_support(yte, ypr, labels=sorted_labels.keys(), average=None)
                            #for k in sorted_labels.keys():
                            #    _ex.add_performance(MEXPerformance(k, P[k], R[k], F[k], 0.0, S[k]))
                            #_conf.add_execution(_ex)
                            #_meta.add_configuration(_conf)
                            # --------------------------------------------------------------------------------------------------------------------------

                        if runLSTM is True:
                            Xtr, Xte, ytr, yte = train_test_split(X1_lstm, y1_lstm, test_size=ds_test_size, random_state=42)  # 352|1440
                            run_lstm(Xtr, Xte, ytr, yte, max_features_1, max_features_2, out_size_1, embedding_size, hidden_size, batch_size, epochs, verbose, maxlen_1)
                else:
                    # ---------------------------------------------------------- META ----------------------------------------------------------
                    #_conf = MEXConfiguration(id=len(_meta.configurations) + 1, horus_enabled=int(horus_feat),
                    #                         dataset_train=ds1[0], dataset_test=ds2[0] ,features=ds1[1], cross_validation=0)
                    # --------------------------------------------------------------------------------------------------------------------------
                    if runDT is True:
                        m = _dt.fit(X1_token, Y1_token)
                        ypr = m.predict(X2_token)
                        #print(skmetrics.classification_report(Y2_token , ypr, labels=PLO_KLASSES.keys(), target_names=PLO_KLASSES.values(), digits=3))
                        P, R, F, S = sklearn.metrics.precision_recall_fscore_support(Y2_token,
                                                                                     np.array(ypr).astype(int),
                                                                                     labels=definitions.PLO_KLASSES.keys())
                        for k in range(len(P)):
                            out_file.write(line % ('False', str(f_key), '1',
                                                   definitions.PLO_KLASSES.get(k + 1),
                                                   P[k], R[k], F[k], str(S[k]), 'DT', ds1_name, ds2_name))
                        # ---------------------------------------------------------- META ----------------------------------------------------------
                        #_ex = MEXExecution(id=len(_conf.executions) + 1, alg='DT', phase='test', random_state=r[d])
                        #P, R, F, S = sklearn.metrics.precision_recall_fscore_support(ds2[1][3] , ypr,
                        #                                                             labels=sorted_labels.keys(),
                        #                                                             average=None)
                        #for k in sorted_labels.keys():
                        #    _ex.add_performance(MEXPerformance(k, P[k], R[k], F[k], 0.0, S[k]))
                        #_conf.add_execution(_ex)
                        #_meta.add_configuration(_conf)
                        # --------------------------------------------------------------------------------------------------------------------------

                    if runCRF is True:
                        print('--CRF')
                        m = _crf.fit(X1_crf, Y1_sentence)
                        ypr = m.predict(X2_crf)
                        #print(metrics.flat_classification_report(ds2[1][1], ypr, labels=sorted_labels.keys(),
                        #                                         target_names=sorted_labels.values(), digits=3))

                        _yte = np.array([tag for row in Y2_sentence for tag in row])

                        P, R, F, S = sklearn.metrics.precision_recall_fscore_support(_yte, _ypr, labels=definitions.PLO_KLASSES.keys())
                        for k in range(len(P)):
                            out_file.write(line % ('False', str(f_key), '1',
                                                   definitions.PLO_KLASSES.get(k + 1),
                                                   P[k], R[k], F[k], str(S[k]), 'CRF', ds1_name, ds2_name))

                    if runLSTM is True:
                        print('--LSTM')
                        max_of_sentences = max(maxlen_1, maxlen_2)
                        X2_lstm = pad_sequences(X2_lstm, maxlen=max_of_sentences)
                        y2_lstm = pad_sequences(y2_lstm, maxlen=max_of_sentences)
                        run_lstm(X1_lstm, X2_lstm, y1_lstm, y2_lstm, max_features_1, max_features_2, out_size_1, embedding_size, hidden_size, batch_size, epochs, verbose, max_of_sentences)

                    if runSTANFORD_NER is True:
                        print('--STANFORD_NER')
                        print(metrics.flat_classification_report(ds2[1][3], ds2[1][2][:11], labels=sorted_labels.keys(), target_names=sorted_labels.values(), digits=3))

    out_file.close()
    #import pickle
    #with open(_label + '.meta', 'wb') as handle:
    #    pickle.dump(_meta, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    parser = argparse.ArgumentParser(
        description='Creates a benchmark pipeline for different classifiers /datasets comparing performance *with* '
                    'and *without* the HORUS features list',
        prog='benchmarking.py',
        usage='%(prog)s [options]',
        epilog='http://horus-ner.org')

    parser.add_argument('--ds', '--datasets', nargs='+', default='2016.conll.freebase.ascii.txt.horus emerging.test.annotated.horus ner.txt.horus 2015.conll.freebase.horus', help='the horus datasets files: e.g.: ritter.horus wnut15.horus')
    #parser.add_argument('--ds', '--datasets', nargs='+', default='test.horus')
    #parser.add_argument('--ds', '--datasets', nargs='+',
    #                    default='test.horus 2015.conll.freebase.horus.short',
    #                    help='the horus datasets files: e.g.: ritter.horus wnut15.horus')
    parser.add_argument('--exp', '--experiment_folder', action='store_true', required=False, help='the sub-folder name where the input file is located', default='EXP_004')
    parser.add_argument('--dt', '--rundt', action='store_true', required=False, default=0, help='benchmarks DT')
    parser.add_argument('--crf', '--runcrf', action='store_true', required=False, default=1, help='benchmarks CRF')
    parser.add_argument('--lstm', '--runlstm', action='store_true', required=False, default=0, help='benchmarks LSTM')
    parser.add_argument('--stanford', '--runstanford', action='store_true', required=False, default=0, help='benchmarks Stanford NER')

    parser.print_help()
    args = parser.parse_args()
    time.sleep(1)

    try:
        benchmark(experiment_folder=args.exp, datasets=args.ds,
                  runCRF=bool(args.crf), runDT=bool(args.dt), runLSTM=bool(args.lstm), runSTANFORD_NER=bool(args.stanford))
    except:
        raise

if __name__ == "__main__":
    main()
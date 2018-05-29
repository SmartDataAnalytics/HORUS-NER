import argparse
import os

import time

import sklearn
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder

from src.classifiers.experiment_metadata import MEXExecution, \
    MEXPerformance, MEX, MEXConfiguration
from src.core.util.definitions import encoder_le1_name

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import matplotlib.pyplot as plt
import sklearn_crfsuite
from sklearn import ensemble
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
from nltk import LancasterStemmer, re, WordNetLemmatizer
import pandas as pd
import pickle
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

def convert_lstm_shape(X, y, horus_feat = False):

    all_text = [c[0] for x in X for c in x]
    all_text.extend([c[1] for x in X for c in x])

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

def shape_data((file, path, le, dict_brown_c1000, dict_brown_c640, dict_brown_c320)):
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
        wnl = WordNetLemmatizer()
        stemmer = SnowballStemmer("english")

        fullpath=path+file
        config.logger.info('reading horus features file: ' + fullpath)
        df = pd.read_csv(fullpath, delimiter="\t", skiprows=1, header=None, keep_default_na=False, na_values=['_|_'])
        df=df.drop(df[df[definitions.INDEX_IS_COMPOUND]==1].index)
        oldsentid = df.iloc[0].at[definitions.INDEX_ID_SENTENCE]
        df = df.reset_index(drop=True)
        COLS = len(df.columns)

        df=pd.concat([df, pd.DataFrame(columns=range(COLS, (COLS + definitions.STANDARD_FEAT_LEN)))], axis=1)
        config.logger.info(len(df))
        for row in df.itertuples():
            index=row.Index
            if index==8:
                a=1
            if index % 500 == 0: config.logger.info(index)
            #print(file + ' - ' + str(index))
            if df.loc[index, definitions.INDEX_ID_SENTENCE] != oldsentid:
                ds_sentences.append(_sent_temp_feat)
                _sent_temp_feat = []
                y_sentences_shape.append(_sent_temp_y)
                _sent_temp_y = []

            idsent = df.loc[index].at[definitions.INDEX_ID_SENTENCE]
            token = df.loc[index].at[definitions.INDEX_TOKEN]
            token=token.decode('utf8', 'ignore')

            lemma=''
            stem=''
            try:
                lemma = wnl.lemmatize(token.lower())
            except: pass
            try:
                stem = stemmer.stem(token.lower())
            except: pass
            brown_1000_path = '{:<016}'.format(dict_brown_c1000.get(token, '0000000000000000'))
            brown_640_path = '{:<016}'.format(dict_brown_c640.get(token, '0000000000000000'))
            brown_320_path = '{:<016}'.format(dict_brown_c320.get(token, '0000000000000000'))

            brown_1000=[]
            k=1
            tot_slide=5 #range(len(brown_1000_path)-1)
            for i in range(tot_slide):
                brown_1000.append(brown_1000_path[:k])
                k+=1
            brown_640 = []
            k = 1
            for i in range(tot_slide):
                brown_640.append(brown_640_path[:k])
                k += 1
            brown_320 = []
            k = 1
            for i in range(tot_slide):
                brown_320.append(brown_320_path[:k])
                k += 1

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
                prev_hyphen = int('-' in prev_token)
                prev_sh = shape(prev_token)
                try: prev_lemma = wnl.lemmatize(prev_token.lower())
                except: prev_lemma = ''.decode('utf8')
                try: prev_stem = stemmer.stem(prev_token.lower())
                except: prev_stem = ''.decode('utf8')
            else:
                prev_pos = ''
                prev_pos_uni = ''
                prev_token = ''
                prev_lemma = ''
                prev_stem = ''
                prev_one_char_token = -1
                prev_special_char = -1
                prev_first_capitalized = -1
                prev_capitalized = -1
                prev_title = -1
                prev_digit = -1
                prev_stop_words = -1
                prev_small = -1
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
                prev_prev_hyphen = int('-' in prev_prev_token)
                prev_prev_sh = shape(prev_prev_token)
                try: prev_prev_lemma = wnl.lemmatize(prev_prev_token.lower())
                except: prev_prev_lemma = ''
                try: prev_prev_stem = stemmer.stem(prev_prev_token.lower())
                except: prev_prev_stem = ''
            else:
                prev_prev_pos= ''
                prev_prev_pos_uni = ''
                prev_prev_token= ''
                prev_prev_lemma=''
                prev_prev_stem=''
                prev_prev_one_char_token= -1
                prev_prev_special_char= -1
                prev_prev_first_capitalized= -1
                prev_prev_capitalized= -1
                prev_prev_title= -1
                prev_prev_digit= -1
                prev_prev_stop_words= -1
                prev_prev_small= -1
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
                next_hyphen = int('-' in next_token)
                next_sh = shape(next_token)
                try: next_lemma = wnl.lemmatize(next_token.lower())
                except: next_lemma = ''
                try: next_stem = stemmer.stem(next_token.lower())
                except: next_stem = ''
            else:
                next_pos = ''
                next_pos_uni = ''
                next_token = ''
                next_lemma=''
                next_stem=''
                next_one_char_token = -1
                next_special_char = -1
                next_first_capitalized = -1
                next_capitalized = -1
                next_title = -1
                next_digit = -1
                next_stop_words = -1
                next_small = -1
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
                next_next_hyphen = int('-' in next_next_token)
                next_next_sh = shape(next_next_token)
                try: next_next_lemma = wnl.lemmatize(next_next_token.lower())
                except: next_next_lemma = ''
                try: next_next_stem = stemmer.stem(next_next_token.lower())
                except: next_next_stem = ''
            else:
                next_next_pos = ''
                next_next_pos_uni = ''
                next_next_token = ''
                next_next_lemma=''
                next_next_stem=''
                next_next_one_char_token = -1
                next_next_special_char = -1
                next_next_first_capitalized = -1
                next_next_capitalized = -1
                next_next_title = -1
                next_next_digit = -1
                next_next_stop_words = -1
                next_next_small = -1
                next_next_hyphen = -1
                next_next_sh = -1

            # standard features [t-2, t-1, t, t+1, t+2] (12*5=60)
            _t.extend([le.transform(prev_prev_pos),
                       le.transform(prev_prev_pos_uni),
                       prev_prev_one_char_token,
                       prev_prev_special_char,
                       prev_prev_first_capitalized,
                       prev_prev_capitalized,
                       prev_prev_title,
                       prev_prev_digit,
                       prev_prev_stop_words,
                       prev_prev_small,
                       prev_prev_hyphen,
                       prev_prev_sh])
            _t.extend([le.transform(prev_pos),
                       le.transform(prev_pos_uni),
                       prev_one_char_token,
                       prev_special_char,
                       prev_first_capitalized,
                       prev_capitalized,
                       prev_title,
                       prev_digit,
                       prev_stop_words,
                       prev_small,
                       prev_hyphen,
                       prev_sh])
            _t.extend([le.transform(df.loc[index].at[definitions.INDEX_POS]),
                       le.transform(df.loc[index].at[definitions.INDEX_POS_UNI]),
                       int(len(token) == 1),
                       int(len(re.findall('(http://\S+|\S*[^\w\s]\S*)', token)) > 0),
                       int(token[0].isupper()),
                       int(token.isupper()),
                       int(token.istitle()),
                       int(token.isdigit()),
                       int(token in stop),
                       int(len(token) <= 2),
                       int('-' in token),
                       shape(token)])
            _t.extend([le.transform(next_pos),
                       le.transform(next_pos_uni),
                       next_one_char_token,
                       next_special_char,
                       next_first_capitalized,
                       next_capitalized,
                       next_title,
                       next_digit,
                       next_stop_words,
                       next_small,
                       next_hyphen,
                       next_sh])
            _t.extend([le.transform(next_next_pos),
                       le.transform(next_next_pos_uni),
                       next_next_one_char_token,
                       next_next_special_char,
                       next_next_first_capitalized,
                       next_next_capitalized,
                       next_next_title,
                       next_next_digit,
                       next_next_stop_words,
                       next_next_small,
                       next_next_hyphen,
                       next_next_sh])

            _t.extend(_append_word_lemma_stem(prev_prev_token.lower(), prev_prev_lemma, prev_prev_stem))
            _t.extend(_append_word_lemma_stem(prev_token.lower(), prev_lemma, prev_stem))
            _t.extend(_append_word_lemma_stem(token.lower(), lemma, stem))
            _t.extend(_append_word_lemma_stem(next_token.lower(), next_lemma, next_stem))
            _t.extend(_append_word_lemma_stem(next_next_token.lower(), next_next_lemma, next_next_stem))
            # word, lemma, stem for [t-2, t-1, t, t+1, t+2] (3*5=15)

            # brown clusters [320, 640, 1000] (5*3=15)
            _t.extend(brown_320)
            _t.extend(brown_640)
            _t.extend(brown_1000)


            #if len(f_indexes) !=0:
            #    _t.extend(df.loc[index][f_indexes])
            df.iloc[[index], COLS:(COLS + definitions.STANDARD_FEAT_LEN)] = _t

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
    except Exception as e:
        print(e)
        raise


def _append_word_lemma_stem(w, l, s):
    t=[]
    try: t.append(enc_word.transform(str(w)))
    except: t.append(0)

    try: t.append(enc_lemma.transform(l.decode('utf-8')))
    except: t.append(0)

    try: t.append(enc_stem.transform(s.decode('utf-8')))
    except: t.append(0)

    return t


def shape_datasets(experiment_folder, datasets):
    ret = []
    job_args = []
    le = joblib.load(config.dir_encoders + encoder_le1_name)
    dict_brown_c1000 = joblib.load(config.dir_datasets + 'gha.500M-c1000-p1.paths_dict.pkl')
    dict_brown_c640 = joblib.load(config.dir_datasets + 'gha.64M-c640-p1.paths_dict.pkl')
    dict_brown_c320 =  joblib.load(config.dir_datasets + 'gha.64M-c320-p1.paths_dict.pkl')
    for ds in datasets:
        config.logger.info(ds)
        _file = config.dir_output + experiment_folder + '_' + ds + '_shaped.pkl'
        if os.path.isfile(_file):
            with open(_file, 'rb') as input:
                shaped = pickle.load(input)
                ret.append(shaped)
        else:
            job_args.append((ds, config.dir_output + experiment_folder, le, dict_brown_c1000, dict_brown_c640, dict_brown_c320))

    config.logger.info('job args created - raw datasets: ' + str(len(job_args)))
    if len(job_args) > 0:
        p = multiprocessing.Pool(8)
        asyncres = p.map(shape_data, job_args)
        config.logger.info(len(asyncres))
        for data in asyncres:
            _file = config.dir_output + experiment_folder + '_' + data[0] + '_shaped.pkl'
            with open(_file, 'wb') as output:
                pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)
            ret.append(data)
            config.logger.info('file exported: ' + _file)
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
    #dfret=df.copy()
    if isinstance(df, pd.DataFrame) == False:
        df=pd.DataFrame(df)
    for icol in df.columns:
        if icol not in f_indexes:
            df.drop(icol, axis=1, inplace=True)
    return df

def get_subset_file_name((_file_path, _file_name, ds, f_key, f_indexes)):
    try:
        config.logger.info(_file_name + ' dump creation starts!')
        X_sentence = [exclude_columns(s, f_indexes) for s in ds[1][0]]
        Y_sentence = [sent2label(s) for s in ds[1][1]]
        X_token = exclude_columns(ds[2][0], f_indexes)
        X_token.replace('O', 0, inplace=True)
        Y_token = [definitions.KLASSES2[y] for y in ds[2][1]]
        X_crf = [sent2features(s) for s in X_sentence]
        _Y_sentence = np.array([x for s in Y_sentence for x in s])  # trick for scikit_learn on CRF (for the precision_recall_fscore_support method)
        #X_lstm, y_lstm, max_features, out_size, maxlen = convert_lstm_shape(X_sentence, Y_sentence, f_indexes)
        #X2_lstm, y2_lstm, max_features_2, out_size_2, maxlen_2 = convert_lstm_shape(ds2[1][0], ds2[1][1], f_indexes)
        #X1_lstm = pad_sequences(X1_lstm, maxlen=max(maxlen_1, maxlen_2))
        #y1_lstm = pad_sequences(y1_lstm, maxlen=max(maxlen_1, maxlen_2))
        with open(_file_path, 'wb') as output:
            pickle.dump([_file_name, f_key, X_sentence, Y_sentence, X_token, Y_token,
                         X_crf, _Y_sentence], output, pickle.HIGHEST_PROTOCOL)
        config.logger.info(_file_name + ' created!')
    except:
        raise

    return _file_name

def benchmark(experiment_folder, datasets, runCRF = False, runDT = False, runLSTM = False, runSTANFORD_NER = False):

    config.logger.info('models: CRF=%s, DT=%s, LSTM=%s, Stanford=%s' % (str(runCRF), str(runDT), str(runLSTM), str(runSTANFORD_NER)))
    experiment_folder+='/'
    config.logger.info('datasets: ' + str(datasets))
    datasets=datasets.split()

    #sorted_labels = definitions.KLASSES.copy()
    #del sorted_labels[4]
    sorted_labels={'PER': 'PER', 'ORG': 'ORG', 'LOC': 'LOC'}
    r = [42, 39, 10, 5, 50]
    # hyper-parameters
    _crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.088, c2=0.002, max_iterations=100, all_possible_transitions=True)
    _crf2 = sklearn_crfsuite.CRF(algorithm='pa', all_possible_transitions=True)
    _dt = ensemble.RandomForestClassifier(n_estimators=50)
    embedding_size = 128
    hidden_size = 32
    batch_size = 128
    epochs = 50
    verbose = 0

    #_meta = MEX('HORUS_EMNLP', _label, 'meta and multi-level machine learning for NLP')

    dict_exp_feat = {1: definitions.FEATURES_STANDARD,
                     2: definitions.FEATURES_STANDARD_WORD,
                     3: definitions.FEATURES_STANDARD_BROWN_64M_c320,
                     4: definitions.FEATURES_STANDARD_BROWN_64M_c640,
                     5: definitions.FEATURES_STANDARD_BROWN_500M_c1000,
                     6: definitions.FEATURES_STANDARD +
                        definitions.FEATURES_STANDARD_WORD +
                        definitions.FEATURES_STANDARD_BROWN_64M_c320,
                     7: definitions.FEATURES_STANDARD +
                        definitions.FEATURES_STANDARD_WORD +
                        definitions.FEATURES_STANDARD_BROWN_64M_c640,
                     8: definitions.FEATURES_STANDARD +
                        definitions.FEATURES_STANDARD_WORD +
                        definitions.FEATURES_STANDARD_BROWN_500M_c1000,
                     9: definitions.FEATURES_HORUS_BASIC_CV,
                     10: definitions.FEATURES_HORUS_BASIC_TX,
                     11: definitions.FEATURES_HORUS_CNN_CV,
                     12: definitions.FEATURES_HORUS_CNN_TX,
                     13: definitions.FEATURES_HORUS_EMB_TX,
                     14: definitions.FEATURES_HORUS_STATS_TX,
                     15: definitions.FEATURES_HORUS_TX,
                     16: definitions.FEATURES_HORUS_TX_EMB,
                     17: definitions.FEATURES_HORUS_CV,
                     18: definitions.FEATURES_HORUS_BASIC_AND_CNN,
                     19: definitions.FEATURES_HORUS,
                     20: definitions.FEATURES_HORUS_BASIC_CV_BEST_STANDARD,
                     21: definitions.FEATURES_HORUS_BASIC_TX_BEST_STANDARD,
                     22: definitions.FEATURES_HORUS_CNN_CV_BEST_STANDARD,
                     23: definitions.FEATURES_HORUS_CNN_TX_BEST_STANDARD,
                     24: definitions.FEATURES_HORUS_EMB_TX_BEST_STANDARD,
                     25: definitions.FEATURES_HORUS_STATS_TX_BEST_STANDARD,
                     26: definitions.FEATURES_HORUS_TX_BEST_STANDARD,
                     27: definitions.FEATURES_HORUS_TX_EMB_BEST_STANDARD,
                     28: definitions.FEATURES_HORUS_CV_BEST_STANDARD,
                     29: definitions.FEATURES_HORUS_BASIC_AND_CNN_BEST_STANDARD,
                     30: definitions.FEATURES_HORUS_BEST_STANDARD}

    config.logger.info('shaping the datasets...')
    shaped_datasets = shape_datasets(experiment_folder, datasets) # ds_name, (X1, y1 [DT-shape]), (X2, y2 [CRF-shape]), (X3, y3 [NN-shape])
    config.logger.info('done! running experiment configurations')
    header = 'cross-validation\tconfig\trun\tlabel\tprecision\trecall\tf1\tsupport\talgo\tdataset1\tdataset2\n'
    line = '%s\t%s\t%s\t%s\t%.5f\t%.5f\t%.5f\t%s\t%s\t%s\t%s\n'

    config.logger.info('creating the sets for all configurations x datasets')
    # multithread to shape the datasets for all configurations in parallel
    job_args = []

    _SET_MASK = '_%s_config_%s.pkl'
    set_file_dump_names=[]
    for ds in shaped_datasets:
        for f_key, f_indexes in dict_exp_feat.iteritems():
            _set_name = _SET_MASK % (ds[0], str(f_key))
            _file = config.dir_output + experiment_folder + _set_name
            if os.path.isfile(_file) is False:
                job_args.append((_file, _set_name, ds, f_key, f_indexes))
            set_file_dump_names.append(_set_name)

    config.logger.info('job args created - config dumps: ' + str(len(job_args)))
    if len(job_args)>0:
        config.logger.info('creating dump files...')
        p = multiprocessing.Pool(8)
        p.map(get_subset_file_name, job_args)
    config.logger.info('done! running the benchmark...')

    # benchmark starts
    name='metadata_'
    if runCRF:
        name+='crf_'
    if runDT:
        name+='dt_'
    if runLSTM:
        name+='lstm_'
    assert name != 'metadata_'
    name += str(dict_exp_feat.keys()).replace('[','').replace(']','').replace(',','').replace(' ','')
    name +='.txt'
    out_file = open(config.dir_output + experiment_folder + name, 'w+')
    out_file.write(header)
    for f_key, f_indexes in dict_exp_feat.iteritems():
        if f_key <= 8:
            config.logger.info('configuration 6 and 7 are the better scenarios - start from 9')
            continue
        for ds1 in datasets:
            ds1_name = ds1
            _set_name = _SET_MASK % (ds1, str(f_key))
            _file = config.dir_output + experiment_folder + _set_name
            config.logger.info('loading [%s]: %s' % (ds1_name, _file))
            with open(_file, 'rb') as input:
                shaped = pickle.load(input)
                ds1_config_name = shaped[0]
                ds1_key = shaped[1]
                X1_sentence = shaped[2]
                Y1_sentence = shaped[3]
                X1_token = shaped[4]
                Y1_token = shaped[5]
                X1_crf = shaped[6]
                _Y1_sentence = shaped[7]
            #pca = PCA(n_components=50)
            #X1_token_PCA = pca.fit(X1_token)
            for ds2 in datasets:
                ds2_name = ds2
                config.logger.info('%s -> %s' % (ds1_name, ds2_name))
                if ds1_name != ds2_name:
                    _set_name = _SET_MASK % (ds2_name, str(f_key))
                    _file = config.dir_output + experiment_folder + _set_name
                    config.logger.info('loading [%s]: %s' % (ds2_name, _file))
                    with open(_file, 'rb') as input:
                        shaped = pickle.load(input)
                        ds2_config_name = shaped[0]
                        ds2_key = shaped[1]
                        X2_sentence = shaped[2]
                        Y2_sentence = shaped[3]
                        X2_token = shaped[4]
                        Y2_token = shaped[5]
                        X2_crf = shaped[6]
                        _Y2_sentence = shaped[7]

                    # ---------------------------------------------------------- META ----------------------------------------------------------
                    # _conf = MEXConfiguration(id=len(_meta.configurations) + 1, horus_enabled=int(horus_feat),
                    #                         dataset_train=ds1[0], dataset_test=ds2[0] ,features=ds1[1], cross_validation=0)
                    # --------------------------------------------------------------------------------------------------------------------------
                    if runDT is True:
                        m = _dt.fit(X1_token, Y1_token)
                        ypr = m.predict(X2_token)
                        # print(skmetrics.classification_report(Y2_token , ypr, labels=PLO_KLASSES.keys(), target_names=PLO_KLASSES.values(), digits=3))
                        P, R, F, S = sklearn.metrics.precision_recall_fscore_support(Y2_token,
                                                                                     np.array(ypr).astype(int),
                                                                                     labels=definitions.PLO_KLASSES.keys())
                        for k in range(len(P)):
                            out_file.write(line % ('False', str(f_key), '1',
                                                   definitions.PLO_KLASSES.get(k + 1),
                                                   P[k], R[k], F[k], str(S[k]), 'DT', ds1_name, ds2_name))
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
                        #print(metrics.flat_classification_report(Y2_sentence, ypr, labels=sorted_labels.keys(),
                        #                                         target_names=sorted_labels.values(), digits=3))

                        _ypr = np.array([tag for row in ypr for tag in row])
                        P, R, F, S = sklearn.metrics.precision_recall_fscore_support(_Y2_sentence, _ypr,
                                                                                     labels=definitions.PLO_KLASSES.values())
                        for k in range(len(P)):
                            out_file.write(line % ('False', str(f_key), '1',
                                                   definitions.PLO_KLASSES.get(k + 1),
                                                   P[k], R[k], F[k], str(S[k]), 'CRF', ds1_name, ds2_name))

                    #if runLSTM is True:
                        #max_of_sentences = max(maxlen_1, maxlen_2)
                        #X2_lstm = pad_sequences(X2_lstm, maxlen=max_of_sentences)
                        #y2_lstm = pad_sequences(y2_lstm, maxlen=max_of_sentences)
                        #run_lstm(X1_lstm, X2_lstm, y1_lstm, y2_lstm, max_features_1, max_features_2, out_size_1,
                        #         embedding_size, hidden_size, batch_size, epochs, verbose, max_of_sentences)

                else:
                    # ---------------------------------------------------------- META ----------------------------------------------------------
                    # _conf = MEXConfiguration(id=len(_meta.configurations)+1, horus_enabled=int(horus_feat),
                    #                                dataset_train=ds1[0], dataset_test=ds1[0], dataset_validation=None, features=None, cross_validation=1)
                    # --------------------------------------------------------------------------------------------------------------------------
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
                                                                                         labels=definitions.PLO_KLASSES.keys())
                            for k in range(len(P)):
                                out_file.write(
                                    line % ('True', str(f_key), str(d + 1), definitions.PLO_KLASSES.get(k + 1),
                                            P[k], R[k], F[k], str(S[k]), 'DT', ds1_name, ds2_name))

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
                            Xtr, Xte, ytr, yte = train_test_split(X1_crf, Y1_sentence, test_size=ds_test_size,
                                                                  random_state=r[d])
                            m = _crf.fit(Xtr, ytr)
                            ypr = m.predict(Xte)

                            _ypr = np.array([tag for row in ypr for tag in row])
                            _yte = np.array([tag for row in yte for tag in row])

                            #print(metrics.flat_classification_report(yte, ypr, labels=sorted_labels.keys(), target_names=sorted_labels.values(), digits=3))

                            P, R, F, S = sklearn.metrics.precision_recall_fscore_support(_yte, _ypr,
                                                                                         labels=definitions.PLO_KLASSES.values())
                            for k in range(len(P)):
                                out_file.write(line % (
                                    'True', str(f_key), str(d + 1), definitions.PLO_KLASSES.get(k + 1), P[k], R[k],
                                    F[k],
                                    str(S[k]), 'CRF', ds1_name, ds2_name))

                            # ---------------------------------------------------------- META ----------------------------------------------------------
                            # _ex = MEXExecution(id=len(_conf.executions)+1, model='', alg='CRF', phase='test', random_state=r[d])
                            # P, R, F, S = sklearn.metrics.precision_recall_fscore_support(yte, ypr, labels=sorted_labels.keys(), average=None)
                            # for k in sorted_labels.keys():
                            #    _ex.add_performance(MEXPerformance(k, P[k], R[k], F[k], 0.0, S[k]))
                            # _conf.add_execution(_ex)
                            # _meta.add_configuration(_conf)
                            # --------------------------------------------------------------------------------------------------------------------------

                        #if runLSTM is True:
                            #Xtr, Xte, ytr, yte = train_test_split(X1_lstm, y1_lstm, test_size=ds_test_size,
                            #                                      random_state=42)  # 352|1440
                            #run_lstm(Xtr, Xte, ytr, yte, max_features_1, max_features_2, out_size_1, embedding_size,
                            #         hidden_size, batch_size, epochs, verbose, maxlen_1)

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

    parser.add_argument('--ds', '--datasets', nargs='+', default='2015.conll.freebase.horus 2016.conll.freebase.ascii.txt.horus ner.txt.horus emerging.test.annotated.horus', help='the horus datasets files: e.g.: ritter.horus wnut15.horus')
    #parser.add_argument('--ds', '--datasets', nargs='+', default='test.horus')
    #parser.add_argument('--ds', '--datasets', nargs='+', default='2015.conll.freebase.horus.short')
    parser.add_argument('--exp', '--experiment_folder', default='EXP_005', action='store_true', required=False, help='the sub-folder name where the input file is located')
    parser.add_argument('--dt', '--rundt', action='store_true', required=False, default=1, help='benchmarks DT')
    parser.add_argument('--crf', '--runcrf', action='store_true', required=False, default=0, help='benchmarks CRF')
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
import argparse
import os

import time

import sklearn

from src.classifiers.experiment_metadata import MEXExecution, \
    MEXPerformance, MEX, MEXConfiguration

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import matplotlib.pyplot as plt
import sklearn_crfsuite
from sklearn import ensemble
from sklearn import metrics as skmetrics
from sklearn.cross_validation import train_test_split
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
ds_test_size = 0.25
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

def get_features(horusfile, le):
    '''
    converts horus features file to algorithm's expected shapes,
    adding further traditional features
    :param horusfile: the horus features file
    :param le: the encoder
    :return: a (standard matrix + a CRF + a LSTM) file formats
    '''
    features, sentence_shape = [], []
    targets, tokens_shape, y_sentences_shape, y_tokens_shape = [], [], [], []

    config.logger.info('reading horus features file: ' + horusfile)
    df = pd.read_csv(horusfile, delimiter="\t", skiprows=1, header=None, keep_default_na=False, na_values=['_|_'])
    oldsentid = df.get_values()[0][definitions.INDEX_ID_SENTENCE]
    for index, feat in df.iterrows():
        if len(feat)>0:
            if feat[definitions.INDEX_IS_COMPOUND] == 0: #no compounds
                if feat[definitions.INDEX_ID_SENTENCE] != oldsentid:
                    sentence_shape.append(features)
                    y_sentences_shape.append(targets)
                    targets, features = [], []

                idsent = feat[definitions.INDEX_ID_SENTENCE]
                idtoken = feat[definitions.INDEX_ID_WORD]

                # standard features

                pos_bef = ''
                pos_aft = ''

                if index > 0 and df.get_value(index-1, definitions.INDEX_IS_COMPOUND) == 0:
                    prev_pos = df.get_value(index-1,definitions.INDEX_POS)

                if index > 1 and df.get_value(index-2, definitions.INDEX_IS_COMPOUND) == 0:
                    prev_prev_pos = df.get_value(index-2, definitions.INDEX_POS)

                if index + 1 < len(df) and df.get_value(index+1, definitions.INDEX_ID_SENTENCE) == idsent \
                        and df.get_value(index+1, definitions.INDEX_IS_COMPOUND) == 0:
                    next_pos = df.get_value(index+1,definitions.INDEX_POS)

                if index + 2 < len(df) and df.get_value(index+2, definitions.INDEX_ID_SENTENCE) == idsent \
                        and df.get_value(index+2, definitions.INDEX_IS_COMPOUND) == 0:
                    next_nex_pos = df.get_value(index+2,definitions.INDEX_POS)

                #TODO: continur aqui
                postag = feat[definitions.INDEX_POS]
                token = feat[definitions.INDEX_TOKEN]
                one_char_token = len(token) == 1
                special_char = len(re.findall('(http://\S+|\S*[^\w\s]\S*)', token)) > 0
                first_capitalized = token[0].isupper()
                capitalized = token.isupper()
                title = token.istitle()
                digit = token.isdigit()
                stop_words = token in stop
                small = True if len(horusfile[3]) <= 2 else False
                lemma = lancaster_stemmer.stem(token)
                hyphen = '-' in token
                sh = shape(token)

                #horus CV features
                cv_basic = feat[11:17]
                cv_cnn = feat[68:83]


                #horus TX features
                tx_basic  = feat[19:25]
                tx_cnn = feat[28:31]
                tx_stats = feat[52:67]



                nr_images_returned = feat[definitions.INDEX_NR_RESULTS_SE_IMG]
                nr_websites_returned = feat[definitions.INDEX_NR_RESULTS_SE_TX]

                cv_loc = float(feat[definitions.INDEX_TOT_CV_LOC])
                cv_org = float(feat[definitions.INDEX_TOT_CV_ORG])
                cv_per = float(feat[definitions.INDEX_TOT_CV_PER])
                cv_dist = float(feat[definitions.INDEX_DIST_CV_I])
                cv_plc = float(feat[definitions.INDEX_PL_CV_I])
                tx_loc = float(feat[definitions.INDEX_TOT_TX_LOC])
                tx_org = float(feat[definitions.INDEX_TOT_TX_ORG])
                tx_per = float(feat[definitions.INDEX_TOT_TX_PER])
                tx_err = float(feat[definitions.INDEX_TOT_ERR_TRANS])
                tx_dist = float(feat[definitions.INDEX_DIST_TX_I])

                if feat[definitions.INDEX_NER] in definitions.NER_TAGS_LOC: ner = u'LOC'
                elif feat[definitions.INDEX_NER] in definitions.NER_TAGS_ORG: ner = u'ORG'
                elif feat[definitions.INDEX_NER] in definitions.NER_TAGS_PER: ner = u'PER'
                else: ner = u'O'

                #standard shape
                f = [idsent, idtoken, token, token.lower(), lemma,
                                pos_bef, postag, pos_aft, definitions.KLASSES2[ner],
                                le.transform(pos_bef), le.transform(postag), le.transform(pos_aft),
                                title, digit, one_char_token, special_char, first_capitalized,
                                hyphen, capitalized, stop_words, small,
                                nr_images_returned, nr_websites_returned,
                                cv_org, cv_loc, cv_per, cv_dist, cv_plc,
                                tx_org, tx_loc, tx_per, tx_dist, tx_err,
                                float(feat[definitions.INDEX_TOT_TX_LOC_TM_CNN]),
                                float(feat[definitions.INDEX_TOT_TX_ORG_TM_CNN]),
                                float(feat[definitions.INDEX_TOT_TX_PER_TM_CNN]),
                                float(feat[definitions.INDEX_DIST_TX_I_TM_CNN]),
                                float(feat[definitions.INDEX_TOT_CV_LOC_1_CNN]),
                                float(feat[definitions.INDEX_TOT_CV_ORG_CNN]),
                                float(feat[definitions.INDEX_TOT_CV_PER_CNN]),
                                float(feat[definitions.INDEX_TOT_CV_LOC_2_CNN]),
                                0 if feat[definitions.INDEX_TOT_CV_LOC_3_CNN] == 'O' else float(feat[definitions.INDEX_TOT_CV_LOC_3_CNN]),
                                0 if feat[definitions.INDEX_TOT_CV_LOC_4_CNN] == 'O' else float(feat[definitions.INDEX_TOT_CV_LOC_4_CNN]),
                                0 if feat[definitions.INDEX_TOT_EMB_SIMILAR_LOC] == 'O' else float(feat[definitions.INDEX_TOT_EMB_SIMILAR_LOC]),
                                0 if feat[definitions.INDEX_TOT_EMB_SIMILAR_ORG] == 'O' else float(feat[definitions.INDEX_TOT_EMB_SIMILAR_ORG]),
                                0 if feat[definitions.INDEX_TOT_EMB_SIMILAR_PER] == 'O' else float(feat[definitions.INDEX_TOT_EMB_SIMILAR_PER]),
                                0 if feat[definitions.INDEX_TOT_CV_LOC_5_CNN] == 'O' else float(feat[definitions.INDEX_TOT_CV_LOC_5_CNN]),
                                0 if feat[definitions.INDEX_TOT_CV_LOC_6_CNN] == 'O' else float(feat[definitions.INDEX_TOT_CV_LOC_6_CNN]),
                                0 if feat[definitions.INDEX_TOT_CV_LOC_7_CNN] == 'O' else float(feat[definitions.INDEX_TOT_CV_LOC_7_CNN]),
                                0 if feat[definitions.INDEX_TOT_CV_LOC_8_CNN] == 'O' else float(feat[definitions.INDEX_TOT_CV_LOC_8_CNN]),
                                0 if feat[definitions.INDEX_TOT_EMB_SIMILAR_NONE] == 'O' else float(feat[definitions.INDEX_TOT_EMB_SIMILAR_NONE]),
                                0 if feat[definitions.INDEX_TOT_TX_NONE_TM_CNN] == 'O' else float(feat[definitions.INDEX_TOT_TX_NONE_TM_CNN])
                                ]

                features.append(f)

                if feat[definitions.INDEX_TARGET_NER] in definitions.NER_TAGS_LOC: y = u'LOC'
                elif feat[definitions.INDEX_TARGET_NER] in definitions.NER_TAGS_ORG: y = u'ORG'
                elif feat[definitions.INDEX_TARGET_NER] in definitions.NER_TAGS_PER: y = u'PER'
                else: y = u'O'

                targets.append(y)

                tokens_shape.append(f[9:len(f)])
                y_tokens_shape.append(definitions.KLASSES2[y])

                oldsentid = feat[1]

    config.logger.info('total of sentences: ' + str(len(sentence_shape)))
    config.logger.info('total of tokens: ' + str(len(tokens_shape)))
    return sentence_shape, y_sentences_shape, tokens_shape, y_tokens_shape

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

def sent2features(sent, horus_feat = False):
    return [features_to_crf_shape(sent, i, horus_feat) for i in range(len(sent))]

def features_to_crf_shape(sent, i, horus_feat):
    "TODO: need to integrate the new features here too!!!!! better to merge the functions!"
    word = sent[i][2]
    postag = sent[i][6]

    features = {
        'bias': 1.0,
        'word.lower()': sent[i][3],
        #'word[-3:]': word[-3:],
        #'word[-2:]': word[-2:],
        'word.isupper()': sent[i][18],
        'word.istitle()': sent[i][12],
        'word.isdigit()': sent[i][13],
        'postag': postag,
        #'postag[:2]': postag[:2],
        'stop_word': sent[i][20],
        'hyphen': sent[i][17],
        'size_small': sent[i][21],
        # 'wordnet_lemmatizer': wordnet_lemmatizer.lemmatize(word),
        'ind1': sent[i][4],
        # 'has_number': hasNumbers(word),
        # 'postag_similar_max': get_similar_words_pos(word)
        # 'gaz_per': True if word in NAMES else False
    }
    if horus_feat is True:
        features.update({
            'ind2': sent[i][23],
            'ind3': sent[i][24],
            'ind4': sent[i][25],
            'ind5': sent[i][26],
            'ind6': sent[i][27],
            'ind7': sent[i][28],
            'ind8': sent[i][29],
            'ind9': sent[i][30],
            'ind10': sent[i][31],
            'ind11': sent[i][32],
        })
    if i > 0:
        word1 = sent[i - 1][2]
        postag1 = sent[i - 1][6]
        features.update({
            '-1:word.lower()': sent[i - 1][3],
            '-1:word.istitle()': sent[i - 1][12],
            '-1:word.isupper()': sent[i - 1][18],
            '-1:postag': postag1,
            #'-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][2]
        postag1 = sent[i + 1][6]
        features.update({
            '+1:word.lower()': sent[i + 1][3],
            '+1:word.istitle()': sent[i + 1][12],
            '+1:word.isupper()': sent[i + 1][12],
            '+1:postag': postag1,
            #'+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features

def get_features_from_datasets(experiment_folder, datasets):
    ret = []
    for ds in datasets:
        ret.append([ds, get_features(config.dir_output + experiment_folder + ds, le1)])
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
    print precision_recall_fscore_support(fyh, fpr)

    pr = model.predict_classes([Xte[:, :, 0], Xte[:, :, 1:Xte.shape[2]]], verbose=verbose)
    yh = yte.argmax(2)
    fyh, fpr = score2(yh, pr)
    print('Testing...')
    print(' - accuracy:', accuracy_score(fyh, fpr))
    print(' - confusion matrix:')
    print(confusion_matrix(fyh, fpr))
    print(' - precision, recall, f1, support:')
    print precision_recall_fscore_support(fyh, fpr)
    print('----------------------------------------------------------------------------------')

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
    _dt = ensemble.RandomForestClassifier(n_estimators=50)
    embedding_size = 128
    hidden_size = 32
    batch_size = 128
    epochs = 50
    verbose = 0

    _label='EXP_003'
    _meta = MEX('HORUS_EMNLP', _label, 'meta and multi-level machine learning for NLP')

    config.logger.info('get the features for each dataset...')
    raw_datasets = get_features_from_datasets(experiment_folder, datasets)
    config.logger.info('done')

    for horus_feat in (False, True):
        print("w/HORUS? " + str(horus_feat))
        for ds1 in raw_datasets:
            if runDT is True:
                X1_dt = ds1[1][2]
            if runCRF is True:
                config.logger.info('shaping to CRF format...')
                X1_crf = [sent2features(s, horus_feat) for s in ds1[1][0]]
            if runLSTM is True:
                X1_lstm, y1_lstm, max_features_1, out_size_1, maxlen_1 = convert_lstm_shape(ds1[1][0], ds1[1][1], horus_feat)
            for ds2 in raw_datasets:
                print("dataset 1 = " + ds1[0])
                print("dataset 2 = " + ds2[0])

                if ds1[0] == ds2[0]:
                    if runDT is True:
                        X2_dt = X1_dt
                    if runCRF is True:
                        X2_crf = X1_crf
                    if runLSTM is True:
                        X2_lstm, y2_lstm, max_features_2, out_size_2, maxlen_2 = convert_lstm_shape(ds2[1][0],
                                                                                                    ds2[1][1],
                                                                                                    horus_feat)
                        X1_lstm = pad_sequences(X1_lstm, maxlen=max(maxlen_1, maxlen_2))
                        y1_lstm = pad_sequences(y1_lstm, maxlen=max(maxlen_1, maxlen_2))
                else:
                    if runDT is True:
                        X2_dt = ds2[1][2]
                    if runCRF is True:
                        X2_crf = [sent2features(s, horus_feat) for s in ds2[1][0]]
                    if runLSTM is True:
                        print('--LSTM')
                        X2_lstm, y2_lstm, max_features_2, out_size_2, maxlen_2 = convert_lstm_shape(ds2[1][0], ds2[1][1], horus_feat)
                        X1_lstm = pad_sequences(X1_lstm, maxlen=max(maxlen_1, maxlen_2))
                        y1_lstm = pad_sequences(y1_lstm, maxlen=max(maxlen_1, maxlen_2))

                # run models
                if ds1[0] == ds2[0]:
                    print("do cross validation")
                    # ---------------------------------------------------------- META ----------------------------------------------------------
                    _conf = MEXConfiguration(id=len(_meta.configurations)+1, horus_enabled=int(horus_feat),
                                                    dataset_train=ds1[0], dataset_test=ds1[0], dataset_validation=None, features=None, cross_validation=1)
                    # --------------------------------------------------------------------------------------------------------------------------
                    for d in range(len(r)):

                        if runCRF is True:
                            print('--CRF')
                            Xtr, Xte, ytr, yte = train_test_split(X1_crf, ds1[1][1], test_size=ds_test_size, random_state=r[d])
                            m = _crf.fit(Xtr, ytr)
                            ypr = m.predict(Xte)
                            print(metrics.flat_classification_report(yte, ypr, labels=sorted_labels.keys(), target_names=sorted_labels.values(), digits=3))

                            # ---------------------------------------------------------- META ----------------------------------------------------------
                            _ex = MEXExecution(id=len(_conf.executions)+1, alg='CRF', phase='test', random_state=r[d])
                            P, R, F, S = sklearn.metrics.precision_recall_fscore_support(yte, ypr, labels=sorted_labels.keys(), average=None)
                            for k in sorted_labels.keys():
                                _ex.add_performance(MEXPerformance(k, P[k], R[k], F[k], 0.0, S[k]))
                            _conf.add_execution(_ex)
                            _meta.add_configuration(_conf)
                            # --------------------------------------------------------------------------------------------------------------------------

                        if runDT is True:
                            print('--DT')
                            X_train = X1_dt
                            if horus_feat == False:
                                X_train = [x[0:12] for x in X1_dt]
                            Xtr, Xte, ytr, yte = train_test_split(X_train, ds1[1][3], test_size=ds_test_size, random_state=r[d])
                            m = _dt.fit(np.array(Xtr).astype(float), np.array(ytr).astype(float))
                            ypr = m.predict(np.array(Xte).astype(float))
                            print(skmetrics.classification_report(yte, ypr, labels=sorted_labels.keys(), target_names=sorted_labels.values(), digits=3))

                            # ---------------------------------------------------------- META ----------------------------------------------------------
                            _ex = MEXExecution(id=len(_conf.executions) + 1, alg='DT', phase='test', random_state=r[d])
                            P, R, F, S = sklearn.metrics.precision_recall_fscore_support(yte, ypr,
                                                                                         labels=sorted_labels.keys(),
                                                                                         average=None)
                            for k in sorted_labels.keys():
                                _ex.add_performance(MEXPerformance(k, P[k], R[k], F[k], 0.0, S[k]))
                            _conf.add_execution(_ex)
                            _meta.add_configuration(_conf)
                            # --------------------------------------------------------------------------------------------------------------------------

                        if runLSTM is True:
                            print('--LSTM')
                            Xtr, Xte, ytr, yte = train_test_split(X1_lstm, y1_lstm, test_size=ds_test_size, random_state=42)  # 352|1440
                            run_lstm(Xtr, Xte, ytr, yte, max_features_1, max_features_2, out_size_1, embedding_size, hidden_size, batch_size, epochs, verbose, maxlen_1)


                else:
                    # ---------------------------------------------------------- META ----------------------------------------------------------
                    _conf = MEXConfiguration(id=len(_meta.configurations) + 1, horus_enabled=int(horus_feat),
                                             dataset_train=ds1[0], dataset_test=ds2[0] ,features=ds1[1], cross_validation=0)
                    # --------------------------------------------------------------------------------------------------------------------------

                    if runCRF is True:
                        print('--CRF')
                        m = _crf.fit(X1_crf, ds1[1][1])
                        ypr = m.predict(X2_crf)
                        print(metrics.flat_classification_report(ds2[1][1], ypr, labels=sorted_labels.keys(), target_names=sorted_labels.values(), digits=3))
                    if runDT is True:
                        print('--DT')
                        X_train = X1_dt
                        X_test = X2_dt
                        if horus_feat == False:
                            X_train = [x[0:12] for x in X1_dt]
                            X_test = [x[0:12] for x in X2_dt]
                        m = _dt.fit(X_train, ds1[1][3])
                        ypr = m.predict(X_test)
                        print(skmetrics.classification_report(ds2[1][3] , ypr, labels=sorted_labels.keys(), target_names=sorted_labels.values(), digits=3))


                        # ---------------------------------------------------------- META ----------------------------------------------------------
                        _ex = MEXExecution(id=len(_conf.executions) + 1, alg='DT', phase='test', random_state=r[d])
                        P, R, F, S = sklearn.metrics.precision_recall_fscore_support(ds2[1][3] , ypr,
                                                                                     labels=sorted_labels.keys(),
                                                                                     average=None)
                        for k in sorted_labels.keys():
                            _ex.add_performance(MEXPerformance(k, P[k], R[k], F[k], 0.0, S[k]))
                        _conf.add_execution(_ex)
                        _meta.add_configuration(_conf)
                        # --------------------------------------------------------------------------------------------------------------------------

                    if runLSTM is True:
                        print('--LSTM')
                        max_of_sentences = max(maxlen_1, maxlen_2)
                        X2_lstm = pad_sequences(X2_lstm, maxlen=max_of_sentences)
                        y2_lstm = pad_sequences(y2_lstm, maxlen=max_of_sentences)
                        run_lstm(X1_lstm, X2_lstm, y1_lstm, y2_lstm, max_features_1, max_features_2, out_size_1, embedding_size, hidden_size, batch_size, epochs, verbose, max_of_sentences)

                    if runSTANFORD_NER is True:
                        print('--STANFORD_NER')
                        print(metrics.flat_classification_report(ds2[1][3], ds2[1][2][:11], labels=sorted_labels.keys(), target_names=sorted_labels.values(), digits=3))

    import pickle
    with open(_label + '.meta', 'wb') as handle:
        pickle.dump(_meta, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    parser = argparse.ArgumentParser(
        description='Creates a benchmark pipeline for different classifiers /datasets comparing performance *with* '
                    'and *without* the HORUS features list',
        prog='benchmarking.py',
        usage='%(prog)s [options]',
        epilog='http://horus-ner.org')

    parser.add_argument('--ds', '--datasets', nargs='+', default='ritter.horus.short.test', help='the horus datasets files: e.g.: ritter.horus wnut15.horus')
    parser.add_argument('--exp', '--experiment_folder', action='store_true', required=False, help='the sub-folder name where the input file is located', default='EXP_003')
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
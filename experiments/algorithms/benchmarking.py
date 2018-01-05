import os

#from experiments.util import horus_to_features
from experiments.util.convert_horusformat_to_conll import horus_to_features

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import matplotlib.pyplot as plt
import sklearn_crfsuite
from nltk import LancasterStemmer
from nltk.corpus import stopwords
from sklearn import ensemble
from sklearn import metrics as skmetrics
from sklearn.cross_validation import train_test_split
from sklearn_crfsuite import metrics

plt.style.use('ggplot')
from horus.core.config import HorusConfig
from horus.core import definitions
import pandas as pd
import re
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
from keras.layers import Embedding, LSTM, Dense, Dropout, Merge

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

ds_test_size = 0.2

def encode(x, n):
    result = np.zeros(n)
    result[x] = 1
    return result
#keras.utils.np_utils.to_categorical or sparse_categorical_crossentropy

def convert_lstm_shape(ds, y, horus_feat = False):
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

def klasses_to_CRF_shape(klasses):
    return klasses

def features_to_crf_shape(sent, i, horus_feat):
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

def shape_datasets():
    print 'shaping datasets...'
    ret = []
    for ds in datasets:
        ret.append([ds[0], horus_to_features(dataset_prefix + ds[0], ds[1])])
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
    print 'compile...'
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

def run_models(runCRF = False, runDT = False, runLSTM = False, runSTANFORD_NER = False):
    if runCRF:
        _crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.088,
            c2=0.002,
            max_iterations=100,
            all_possible_transitions=True
        )
    if runDT:
        _dt = ensemble.RandomForestClassifier(n_estimators=50)
    if runLSTM:
        embedding_size = 128
        hidden_size = 32
        batch_size = 128
        epochs = 50
        verbose = 0

    raw_datasets = shape_datasets()

    for horus_feat in (False, False):
        print "HORUS? ", horus_feat
        for ds1 in raw_datasets:
            if runDT: X1_dt = ds1[1][2]
            if runCRF: X1_crf = [sent2features(s, horus_feat) for s in ds1[1][0]]
            if runLSTM: X1_lstm, y1_lstm, max_features_1, out_size_1, maxlen_1 = convert_lstm_shape(ds1[1][0], ds1[1][1], horus_feat)
            for ds2 in raw_datasets:
                if runDT:
                    print '--DT'
                    X2_dt = ds2[1][2]
                if runCRF:
                    print '--CRF'
                    X2_crf = [sent2features(s, horus_feat) for s in ds2[1][0]]
                if runLSTM:
                    pass
                    print '--LSTM'
                    X2_lstm, y2_lstm, max_features_2, out_size_2, maxlen_2 = convert_lstm_shape(ds2[1][0], ds2[1][1], horus_feat)
                    X1_lstm = pad_sequences(X1_lstm, maxlen=max(maxlen_1, maxlen_2))
                    y1_lstm = pad_sequences(y1_lstm, maxlen=max(maxlen_1, maxlen_2))
                print "---------------------------------------------------"
                print "dataset 1 = ", ds1[0]
                print "dataset 2 = ", ds2[0]
                if ds1[0] == ds2[0]:
                    print "do cross validation"
                    for d in range(len(r)):
                        if runCRF:
                            print '--CRF'
                            Xtr, Xte, ytr, yte = train_test_split(X1_crf, ds1[1][1], test_size=ds_test_size, random_state=r[d])
                            m = _crf.fit(Xtr, ytr)
                            ypr = m.predict(Xte)
                            print(metrics.flat_classification_report(yte, ypr, labels=sorted_labels, digits=3))
                        if runDT:
                            print '--DT'
                            X_train = X1_dt
                            if horus_feat == False:
                                X_train = [x[0:12] for x in X1_dt]
                            Xtr, Xte, ytr, yte = train_test_split(X_train, ds1[1][3], test_size=ds_test_size, random_state=r[d])
                            m = _dt.fit(np.array(Xtr.astype(float), np.array(ytr).astype(float)))
                            ypr = m.predict(np.array(Xte).astype(float))
                            print(skmetrics.classification_report(yte, ypr, labels=sorted_labels , digits=3)) #labels=[1, 2, 3], target_names=['LOC', 'ORG', 'PER']
                        if runLSTM:
                            print '--LSTM'
                            Xtr, Xte, ytr, yte = train_test_split(X1_lstm, y1_lstm, test_size=ds_test_size, random_state=42)  # 352|1440
                            run_lstm(Xtr, Xte, ytr, yte, max_features_1, max_features_2, out_size_1, embedding_size, hidden_size, batch_size, epochs, verbose, maxlen_1)


                else:
                    if runCRF:
                        print '--CRF'
                        m = _crf.fit(X1_crf, ds1[1][1])
                        ypr = m.predict(X2_crf)
                        print(metrics.flat_classification_report(ds2[1][1], ypr, labels=sorted_labels, digits=3))
                    if runDT:
                        print '--DT'
                        X_train = X1_dt
                        X_test = X2_dt
                        if horus_feat == False:
                            X_train = [x[0:12] for x in X1_dt]
                            X_test = [x[0:12] for x in X2_dt]
                        m = _dt.fit(X_train, ds1[1][3])
                        ypr = m.predict(X_test)
                        print(skmetrics.classification_report(ds2[1][3] , ypr, labels=sorted_labels, digits=3))
                    if runLSTM:
                        print '--LSTM'
                        max_of_sentences = max(maxlen_1, maxlen_2)
                        X2_lstm = pad_sequences(X2_lstm, maxlen=max_of_sentences)
                        y2_lstm = pad_sequences(y2_lstm, maxlen=max_of_sentences)
                        run_lstm(X1_lstm, X2_lstm, y1_lstm, y2_lstm, max_features_1, max_features_2, out_size_1, embedding_size, hidden_size, batch_size, epochs, verbose, max_of_sentences)

                    if runSTANFORD_NER:
                        print '--STANFORD_NER'
                        print(metrics.flat_classification_report(ds2[1][3], ds2[1][2][:11], labels=sorted_labels, digits=3))


le1 = joblib.load(config.encoder_path + "_encoder_pos.pkl")
le2 = joblib.load(config.encoder_path + "_encoder_nltk2.pkl")

dataset_prefix = config.output_path + "experiments/EXP_do_tokenization/"
datasets = (("out_exp003_ritter_en_tweetNLP.csv", le1),
            ("out_exp003_wnut15_en_tweetNLP.csv", le1),
            ("out_exp003_wnut16_en_tweetNLP.csv", le1),
            ("out_exp003_coNLL2003testA_en_NLTK.csv", le2))

#labels = list(crf.classes_)
labels = list(['LOC', 'ORG', 'PER'])
#labels.remove('O')
# group B and I results
sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)

r = [42, 39, 10, 5, 50]

run_models(True, False, False, False)

exit(0)


# training

crf2 = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.18687907015736968,
    c2=0.025503200544851036,
    max_iterations=100,
    all_possible_transitions=True
)
#crf2.fit(X_train_CRF_shape, y_train_CRF_shape)

# eval


#y_pred2 = crf2.predict(X_test_CRF_shape)

#metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)



#print(metrics.flat_classification_report(
#    y_test, y_pred2, labels=sorted_labels, digits=3
#))
exit(0)
#r = [42, 39, 10, 5, 50]
#fmeasures = []
#for d in range(len(r)):
#    cv_X_train, cv_X_test, cv_y_train, cv_y_test = train_test_split(X_train_CRF_shape, y_train_CRF_shape,
#                                                        test_size = 0.30, random_state = r[d])
#    m = crf.fit(cv_X_train, cv_y_train)
#    cv_y_pred = m.predict(cv_X_test)
#    print(metrics.flat_classification_report(
#        cv_y_test, cv_y_pred, labels=sorted_labels, digits=3
#    ))
    #cv_y_test_bin = MultiLabelBinarizer().fit_transform(cv_y_test)
    #cv_y_pred_bin = MultiLabelBinarizer().fit_transform(cv_y_pred)
    #fmeasures.append(f1_score(cv_y_test_bin, cv_y_pred_bin, average='weighted'))

#print sum(fmeasures)/len(r)


#scores = cross_val_score(crf, _X, _y, cv=5, scoring='f1_macro')
#scores2 = cross_val_score(crf2, _X, _y, cv=5, scoring='f1_macro')

#rs = ShuffleSplit(n_splits=3, test_size=.20, random_state=0)
#for train_index, test_index in rs.split(_X):
#    print("TRAIN:", train_index, "TEST:", test_index)

#print scores
#print scores2

#exit(0)

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


ax.scatter(_x, _y, c=_c, s=60, alpha=0.9, edgecolors=[0,0,0])
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
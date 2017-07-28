import matplotlib.pyplot as plt
import sklearn_crfsuite
from nltk import LancasterStemmer
from nltk.corpus import stopwords
from sklearn import ensemble
from sklearn import metrics as skmetrics
from sklearn.cross_validation import train_test_split
from sklearn_crfsuite import metrics

plt.style.use('ggplot')
from horus.core import HorusConfig
from horus import definitions
import pandas as pd
import re
from sklearn.externals import joblib
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.grid_search import RandomizedSearchCV
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

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
stop = set(stopwords.words('english'))
lancaster_stemmer = LancasterStemmer()

def horus_to_features(horusfile, le):
    print horusfile
    features, sentences_shape = [], []
    targets, tokens_shape, y_sentences_shape, y_tokens_shape = [], [], [], []

    df = pd.read_csv(horusfile, delimiter=",", skiprows=1, header=None, keep_default_na=False, na_values=['_|_'])
    countaux = 0
    oldsentid = df.get_values()[0][1]
    for index, linha in df.iterrows():
        countaux+=1
        if len(linha)>0:
            if linha[7] == 0: # no compounds
                if linha[1] != oldsentid:
                    sentences_shape.append(features)
                    y_sentences_shape.append(targets)
                    targets, features = [],  []
                else: # same sentence
                    idsent = linha[1]
                    idtoken = linha[2]
                    pos_bef = ''
                    pos_aft = ''
                    if index > 0 and df.get_value(index - 1, 7) == 0:
                        pos_bef = df.get_value(index - 1, 5)
                    if index + 1 < len(df) and df.get_value(index + 1, 7) == 0:
                        pos_aft = df.get_value(index + 1, 5)
                    token = linha[3]
                    postag = linha[5]
                    one_char_token = len(token) == 1
                    special_char = len(re.findall('(http://\S+|\S*[^\w\s]\S*)', token)) > 0
                    first_capitalized = token[0].isupper()
                    capitalized = token.isupper()
                    title = token.istitle()
                    digit = token.isdigit()
                    stop_words = token in stop
                    small =  True if len(horusfile[3]) <= 2 else False
                    stemmer_lanc = lancaster_stemmer.stem(token)
                    nr_images_returned = linha[17]
                    nr_websites_returned = linha[25]
                    hyphen = '-' in token
                    cv_loc = float(linha[12])
                    cv_org = float(linha[13])
                    cv_per = float(linha[14])
                    cv_dist = float(linha[15])
                    cv_plc = float(linha[16])
                    tx_loc = float(linha[20])
                    tx_org = float(linha[21])
                    tx_per = float(linha[22])
                    tx_err = float(linha[23])
                    tx_dist = float(linha[24])

                    if linha[6] in definitions.NER_TAGS_LOC: ner = u"LOC"
                    elif linha[6] in definitions.NER_TAGS_ORG: ner = u"ORG"
                    elif linha[6] in definitions.NER_TAGS_PER: ner = u"PER"
                    else: ner = u"O"

                    # standard shape
                    sel_features = [idsent, idtoken, token, token.lower(), stemmer_lanc,
                                    pos_bef, postag, pos_aft, definitions.KLASSES2[ner],
                                    le.transform(pos_bef), le.transform(postag), le.transform(pos_aft),
                                    title, digit, one_char_token, special_char, first_capitalized,
                                    hyphen, capitalized, stop_words, small,
                                    nr_images_returned, nr_websites_returned,
                                    cv_org, cv_loc, cv_per, cv_dist, cv_plc,
                                    tx_org, tx_loc, tx_per, tx_dist, tx_err]

                    features.append(sel_features)

                    if linha[51] in definitions.NER_TAGS_LOC: y = u"LOC"
                    elif linha[51] in definitions.NER_TAGS_ORG: y = u"ORG"
                    elif linha[51] in definitions.NER_TAGS_PER: y = u"PER"
                    else: y = u"O"

                    targets.append(y)

                    #selected_features = numpy.array(selected_features)
                    #selected_features = np.delete(selected_features, np.s_[0:2], axis=1)

                    tokens_shape.append(sel_features[9:len(sel_features)])
                    y_tokens_shape.append(definitions.KLASSES2[y])

                oldsentid = linha[1]

    print 'total of sentences', len(sentences_shape)
    print 'total of tokens', len(tokens_shape)
    #print set(Y)
    #print set(teste)
    return sentences_shape, y_sentences_shape, tokens_shape, y_tokens_shape

def sent2features(sent, alg, horus_feat = False):
    if alg=='CRF':
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

def run_models(CRF = False, DT = False, LSTM = False, STANFORD_NER = False):
    if CRF:
        _crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.088,
            c2=0.002,
            max_iterations=100,
            all_possible_transitions=True
        )
    if DT:
        _dt = ensemble.RandomForestClassifier(n_estimators=50)
    if LSTM:
        _lstm = Sequential()
        _lstm.add(Dense(12, input_dim=19, kernel_initializer='uniform', activation='relu'))
        _lstm.add(Dense(8, kernel_initializer='uniform', activation='relu'))
        _lstm.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
        _lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    shaped_datasets = shape_datasets()

    for horus_feat in (True, False):
        print "HORUS? ", horus_feat
        for ds1 in shaped_datasets:
            X1_CRF_shape = [sent2features(s, 'CRF', horus_feat) for s in ds1[1][0]]
            for ds2 in shaped_datasets:
                X2_CRF_shape = [sent2features(s, 'CRF', horus_feat) for s in ds2[1][0]]
                print "---------------------------------------------------"
                print "dataset 1 = ", ds1[0]
                print "dataset 2 = ", ds2[0]
                if ds1[0] == ds2[0]:
                    print "do cross validation"
                    for d in range(len(r)):
                        if CRF:
                            print '--CRF'
                            X_train, X_test, y_train, y_test \
                                = train_test_split(X1_CRF_shape, ds1[1][1], test_size=0.30, random_state=r[d])
                            m = _crf.fit(X_train, y_train)
                            y_pred = m.predict(X_test)
                            print(metrics.flat_classification_report(
                                y_test, y_pred, labels=sorted_labels, digits=3)
                            )
                        if DT:
                            if horus_feat == False:
                                X_train = [x[0:12] for x in ds1[1][2]]
                            else:
                                X_train = ds1[1][2]
                            y_train = ds1[1][3]

                            print '--DT'
                            s_X_train, s_X_test, s_y_train, s_y_test \
                                = train_test_split(X_train, y_train, test_size=0.30, random_state=r[d])
                            dtmodel = _dt.fit(np.array(s_X_train).astype(float), np.array(s_y_train).astype(float))
                            y_pred = dtmodel.predict(np.array(s_X_test).astype(float))
                            print(skmetrics.classification_report(
                                s_y_test, y_pred, labels=[1, 2, 3], target_names=['LOC', 'ORG', 'PER'], digits=3)
                            )
                else:
                    if LSTM:
                        print '--LSTM'

                    if CRF:
                        print '--CRF'
                        m = _crf.fit(X1_CRF_shape, ds1[1][1])
                        crf_pred = m.predict(X2_CRF_shape)
                        print(metrics.flat_classification_report(
                            ds2[1][1], crf_pred, labels=sorted_labels, digits=3
                        ))
                    if DT:
                        print '--DT'
                        if horus_feat == False:
                            X_train = [x[0:12] for x in ds1[1][2]]
                            X_test = [x[0:12] for x in ds2[1][2]]
                        else:
                            X_train = ds1[1][2]
                            X_test = ds2[1][2]

                        y_train = ds1[1][3]
                        y_test = ds2[1][3]
                        m = _dt.fit(X_train, y_train)
                        y_pred = m.predict(X_test)
                        print(skmetrics.classification_report(
                            y_test , y_pred, labels=[1, 2, 3], target_names=['LOC', 'ORG', 'PER'], digits=3)
                        )
                    if STANFORD_NER:
                        print '--STANFORD_NER'
                        print(metrics.flat_classification_report(
                            ds2[1][3], ds2[1][2][:11], labels=sorted_labels, digits=3
                        ))


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

run_models(False, True, False, False)

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
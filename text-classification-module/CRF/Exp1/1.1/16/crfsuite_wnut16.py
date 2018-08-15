import nltk
import sklearn_crfsuite
import numpy as np
from sklearn_crfsuite.metrics import flat_classification_report, flatten
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.model_selection import	train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import TweetTokenizer
import  CRF.definitions as definitions
from spacy.tokenizer import Tokenizer
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier


nlp = spacy.load("en_core_web_sm")
tokenizer = Tokenizer(nlp.vocab)

lancaster_stemmer = LancasterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
tknzr = TweetTokenizer(preserve_case=True, strip_handles=False, reduce_len=False)
stop = set(stopwords.words('english'))

def get_tuples(dspath):
    sentences = []
    s = ''
    tokens = []
    ners = []
    poss = []
    tot_sentences = 0
    ners_by_position = []
    index = 0
    with open(dspath) as f:
        for line in f:
            if line.strip() != '':
                token = line.split('\t')[0].decode('utf-8')
                ner = line.split('\t')[1].replace('\r', '').replace('\n', '').decode('utf-8')
                '''
                if ner in definitions.NER_TAGS_ORG:
                    ner = 'ORG'
                elif ner in definitions.NER_TAGS_LOC:
                    ner = 'LOC'
                elif ner in definitions.NER_TAGS_PER:
                    ner = 'PER'
                else :
                    ner = 'O'
                '''
                #ners_by_position.append([index, len(token), ner])
                index += len(token) + 1
            if line.strip() == '':
                if len(tokens) != 0:
                    tknz = tokenizer(s)
                    tknz2 = []
                    for x in tknz:
                        x = str(x).decode('utf-8')
                        if x == u"  ":
                            tknz2.append(u" ")
                            tknz2.append(u" ")
                        else:
                            tknz2.append(x)

                    poss = [x[1].decode('utf-8') for x in nltk.pos_tag(tknz2)]

                    sentences.append(zip(tokens, poss, ners))
                    tokens = []
                    ners = []
                    s = ''
                    tot_sentences += 1


            else:
                s += token + ' '
                tokens.append(token)
                ners.append(ner)

    return sentences

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1].encode("utf-8")
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2].encode("utf-8"),
        'stop_word': word in stop,
        'hyphen': '-' in word,
        'size_small': True if len(word) <= 2 else False,
        'stemmer_lanc': lancaster_stemmer.stem(word),
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1].encode("utf-8")
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2].encode("utf-8"),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1].encode("utf-8")
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2].encode("utf-8"),
        })
    else:
        features['EOS'] = True

    return features

def word2features_new(sent, i):
    word = sent[i][0]
    postag = sent[i][1].encode("utf-8")
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2].encode("utf-8"),
        'stop_word': word in stop,
        'hyphen': '-' in word,
        'size_small': True if len(word) <= 2 else False,
        'stemmer_lanc': lancaster_stemmer.stem(word),
        'klass_1': tf_idf_clone_1.predict([word])[0],
        'klass': tf_idf_clone.predict([word])[0],
        'klass_2': tf_idf_clone_2.predict([word])[0],
        'klass_3': tf_idf_clone_3.predict([word])[0],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1].encode("utf-8")
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2].encode("utf-8"),
            '-1:klass': tf_idf_clone.predict([word])[0],
            '-1:klass_1': tf_idf_clone_1.predict([word])[0],
            '-1:klass_2': tf_idf_clone_2.predict([word])[0],
            '-1:klass_3': tf_idf_clone_3.predict([word])[0],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1].encode("utf-8")
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2].encode("utf-8"),
            '+1:klass': tf_idf_clone.predict([word])[0],
            '+1:klass_1': tf_idf_clone_1.predict([word])[0],
            '+1:klass_2': tf_idf_clone_2.predict([word])[0],
            '+1:klass_3': tf_idf_clone_3.predict([word])[0],
        })
    else:
        features['EOS'] = True

    return features

def group_labels(labels):
    y = []
    for string in labels:
        temp = []
        for tok in string:
            if tok.find("geo-loc") != -1 or tok.find("location") != -1:
                temp.append("LOC")
            elif tok.find("company") != -1 or tok.find("corporation") != -1:
                temp.append("ORG")
            elif tok.find("person") != -1:
                temp.append("PER")
            else:
                temp.append("O")

        y.append(temp)

    return y


def group_labels_to_num(labels):
    y = []
    for string in labels:
        temp = []
        for tok in string:
            if tok.find("B-company"):
                temp.append(1)
            elif tok.find("I-company"):
                temp.append(2)
            elif tok.find("B-facility"):
                temp.append(3)
            elif tok.find("I-facility"):
                temp.append(4)
            elif tok.find("B-geo-loc"):
                temp.append(5)
            elif tok.find("I-geo-loc"):
                temp.append(6)
            elif tok.find("B-movie"):
                temp.append(7)
            elif tok.find("I-movie"):
                temp.append(8)
            elif tok.find("B-musicartist"):
                temp.append(9)
            elif tok.find("I-musicartist"):
                temp.append(10)
            elif tok.find("B-other"):
                temp.append(11)
            elif tok.find("I-other"):
                temp.append(12)
            elif tok.find("B-person"):
                temp.append(13)
            elif tok.find("I-person"):
                temp.append(14)
            elif tok.find("B-product"):
                temp.append(15)
            elif tok.find("I-product"):
                temp.append(16)
            elif tok.find("B-sportsteam"):
                temp.append(17)
            elif tok.find("I-sportsteam"):
                temp.append(18)
            elif tok.find("B-tvshow"):
                temp.append(19)
            elif tok.find("I-tvshow"):
                temp.append(20)
            else:
                temp.append(21)

        y.append(temp)

    return y


def group_num_to_labels(labels):
    y = []
    for string in labels:
        temp = []
        for tok in string:
            if tok.find(1):
                temp.append("B-company")
            elif tok.find(2):
                temp.append("I-company")
            elif tok.find(3):
                temp.append("B-facility")
            elif tok.find(4):
                temp.append("I-facility")
            elif tok.find(5):
                temp.append("B-geo-loc")
            elif tok.find(6):
                temp.append("I-geo-loc")
            elif tok.find(7):
                temp.append("B-movie")
            elif tok.find(8):
                temp.append("I-movie")
            elif tok.find(9):
                temp.append("B-musicartist")
            elif tok.find(10):
                temp.append("I-musicartist")
            elif tok.find(11):
                temp.append("B-other")
            elif tok.find(12):
                temp.append("I-other")
            elif tok.find(13):
                temp.append("B-person")
            elif tok.find(14):
                temp.append("I-person")
            elif tok.find(15):
                temp.append("B-product")
            elif tok.find(16):
                temp.append("I-product")
            elif tok.find(17):
                temp.append("B-sportsteam")
            elif tok.find(18):
                temp.append("I-sportsteam")
            elif tok.find(19):
                temp.append("B-tvshow")
            elif tok.find(20):
                temp.append("I-tvshow")
            else:
                temp.append(21)

        y.append(temp)

    return y


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2features_new(sent):
    return [word2features_new(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label.encode("utf-8") for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

dataset_wnut16_train = get_tuples('../../../../data/test_data/WNUT/16/train.txt')
dataset_wnut16_test = get_tuples('../../../../data/test_data/WNUT/16/test.txt')


train_sents = dataset_wnut16_train

test_sents = dataset_wnut16_test

tf_idf_clone_1 = joblib.load('../../../../one-hot-classifiers/tf-idf+svm_1.pkl')
tf_idf_clone_2 = joblib.load('../../../../one-hot-classifiers/tf-idf+svm_2.pkl')
tf_idf_clone_3 = joblib.load('../../../../one-hot-classifiers/tf-idf+svm_3.pkl')
tf_idf_clone = joblib.load('../../../../multi-class-classifier/tf-idf+svm/tf-idf+svm_new.pkl')

# X_train = [sent2features(s) for s in train_sents]
# X_train_new = [sent2features_new(s) for s in train_sents]
y_train_raw = [sent2labels(s) for s in train_sents]
# #
# X_test = [sent2features(s) for s in test_sents]
# X_test_new = [sent2features_new(s) for s in test_sents]
y_test_raw = [sent2labels(s) for s in test_sents]
#
# y_test = group_labels(y_test_raw)
# y_train = group_labels(y_train_raw)

# joblib.dump(X_train, 'X_train.pkl', compress=9)
# joblib.dump(X_train_new, 'X_train_new.pkl', compress=9)
# joblib.dump(y_train, 'y_train.pkl', compress=9)

# joblib.dump(X_test, 'X_test.pkl', compress=9)
# joblib.dump(y_test, 'y_test.pkl', compress=9)
# joblib.dump(X_test_new, 'X_test_new.pkl', compress=9)
X_train = joblib.load('X_train.pkl')
X_train_new = joblib.load('X_train_new.pkl')
# y_train = joblib.load('y_train.pkl')

X_test = joblib.load('X_test.pkl')
X_test_new = joblib.load('X_test_new.pkl')
# y_test = joblib.load('y_test.pkl')

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.088,
    c2=0.002,
    max_iterations=100,
    all_possible_transitions=True,
)

crf_new = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.088,
    c2=0.002,
    max_iterations=100,
    all_possible_transitions=True,
)

rnd_clf = Pipeline([('vect', DictVectorizer()), ('clf-svm', RandomForestClassifier())])

rnd_clf_new	 = Pipeline([('vect', DictVectorizer()), ('clf-svm', RandomForestClassifier())])

nb_clf = Pipeline([('vect', DictVectorizer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42))])

nb_clf_new = Pipeline([('vect', DictVectorizer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42))])

for i in range(5):
    cur = np.random
    X_train_new = joblib.load('X_train_new.pkl')
    X_test_new = joblib.load('X_test_new.pkl')

    X_train_folds_new, X_test_folds_new, y_train_folds, y_test_folds = X_train_new, X_test_new, y_train_raw, y_test_raw

    #Train CRF
    crf_new.fit(X_train_folds_new, y_train_folds)

    #flatten data for random forest trees
    flat_X_train_folds_new = flatten(X_train_folds_new)
    flat_y_train_folds = flatten(y_train_folds)

    flat_X_test_folds_new = flatten(X_test_folds_new)
    flat_y_test_folds = flatten(y_test_folds)

    #Train random forest treees
    rnd_clf_new.fit(flat_X_train_folds_new, flat_y_train_folds)

    nb_clf_new.fit(flat_X_train_folds_new, flat_y_train_folds)

    #predic using CRF
    y_pred_new = crf_new.predict(X_test_folds_new)

    #predic using Random forest trees
    y_pred_rnd_new = rnd_clf_new.predict(flat_X_test_folds_new)
    y_pred_nb_new = nb_clf_new.predict(flat_X_test_folds_new)

    X_train_folds = []
    X_test_folds = []

    for sentence in X_train_folds_new:
        tmp = []
        for token in sentence:
            del token['klass']
            del token['klass_1']
            del token['klass_2']
            del token['klass_3']

            if '+1:klass' in token:
                del token['+1:klass']
                del token['+1:klass_1']
                del token['+1:klass_2']
                del token['+1:klass_3']

            if '-1:klass' in token:
                del token['-1:klass']
                del token['-1:klass_1']
                del token['-1:klass_2']
                del token['-1:klass_3']
            tmp.append(token)
        X_train_folds.append(tmp)

    for sentence in X_test_folds_new:
            tmp = []
            for token in sentence:

                del token['klass']
                del token['klass_1']
                del token['klass_2']
                del token['klass_3']

                if '+1:klass' in token:
                    del token['+1:klass']
                    del token['+1:klass_1']
                    del token['+1:klass_2']
                    del token['+1:klass_3']

                if '-1:klass' in token:
                    del token['-1:klass']
                    del token['-1:klass_1']
                    del token['-1:klass_2']
                    del token['-1:klass_3']
                tmp.append(token)
            X_test_folds.append(tmp)

    #BASELINE below
    # Train CRF
    crf.fit(X_train_folds, y_train_folds)

    # flatten data for random forest trees
    flat_X_train_folds = flatten(X_train_folds)

    flat_X_test_folds = flatten(X_test_folds)
    # Train random forest treees
    rnd_clf.fit(flat_X_train_folds, flat_y_train_folds)
    nb_clf.fit(flat_X_train_folds, flat_y_train_folds)

    # predic using CRF
    y_pred = crf.predict(X_test_folds)

    # predic using Random forest trees
    y_pred_rnd = rnd_clf.predict(flat_X_test_folds)
    y_pred_nb = nb_clf.predict(flat_X_test_folds)

    labels = list(crf.classes_)
    labels.remove('O')

    print(flat_classification_report(y_test_folds, y_pred, labels=labels, digits=3,
                                     target_names=labels))
    print "------------------- new ---------------------"
    print(flat_classification_report(y_test_folds, y_pred_new, labels=labels, digits=3,
                                     target_names=labels))
    print("####################### Random Forest ########################")
    print(classification_report(flat_y_test_folds, y_pred_rnd, labels=labels, digits=3,
                                     target_names=labels))
    print "------------------- new ---------------------"
    print(classification_report(flat_y_test_folds, y_pred_rnd_new, labels=labels, digits=3,
                                     target_names=labels))
    print("####################### SGD Classifier ########################")
    print(classification_report(flat_y_test_folds, y_pred_nb, labels=labels, digits=3,
                                target_names=labels))
    print "------------------- new ---------------------"
    print(classification_report(flat_y_test_folds, y_pred_nb_new, labels=labels, digits=3,
                                target_names=labels))
    print "*****************************************************************************************"



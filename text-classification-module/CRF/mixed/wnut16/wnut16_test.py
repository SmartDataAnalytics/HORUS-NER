import nltk
import sklearn_crfsuite
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.externals import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import TweetTokenizer
import  CRF.definitions as definitions
from spacy.tokenizer import Tokenizer
from spacy.attrs import ORTH, LEMMA
import spacy
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
                    #poss = [x[1].decode('utf-8') for x in nltk.pos_tag(nltk.word_tokenize(s[:-1]))]

                    tknz = tokenizer(s)
                    tknz2 = []
                    # print len(tknz)
                    for x in tknz:
                        x = str(x).decode('utf-8')
                        if x == u"  ":
                            tknz2.append(u" ")
                            tknz2.append(u" ")
                        else:
                            tknz2.append(x)

                    # print tknz2
                    poss = [x[1].decode('utf-8') for x in nltk.pos_tag(tknz2)]

                    # gold = GoldParse(doc, words=words, tags=tags)

                    # assert len(poss) == len(tokens) == len(ners)

                    sentences.append(zip(tokens, poss, ners))
                    #else:
                    #    aux = 0
                    #    for i in range(len()):
                    #        if aux <= tokens[i]
                    # if len(poss) != len(tokens) or len(poss) != len(ners):
                    #     print (poss)
                    #     print tknz2, len(tknz2)
                    #     print(tokens), len(tokens)
                    #     print(ners)
                    #     print "---------------------------"
                    tokens = []
                    ners = []
                    s = ''
                    tot_sentences += 1


            else:
                s += token + ' '
                tokens.append(token)
                ners.append(ner)

    return sentences


def remove_extra_features(X_train_new, X_test_new):
    train = []
    test = []

    train.extend(X_train_new)
    test.extend(X_test_new)

    X_train = []
    X_test = []

    for sentence in train:
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
        X_train.append(tmp)

    for sentence in test:
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
        X_test.append(tmp)

    return X_train, X_test

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
            if tok.find("B-geo-loc") != -1 or tok.find("B-location") != -1:
                temp.append("LOC")
            else:
                if tok.find("B-company") != -1 or tok.find("B-corporation") != -1:
                    temp.append("ORG")
                else:
                    if tok.find("B-person") != -1:
                        temp.append("PER")
                    else:
                        temp.append("O")

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


dataset_wnut15_train = get_tuples('../../../data/test_data/WNUT/15/2015.conll.freebase')
dataset_wnut16_train = get_tuples('../../../data/test_data/WNUT/16/train.txt')
dataset_wnut16_test = get_tuples('../../../data/test_data/WNUT/16/test.txt')
dataset_wnut17_train = get_tuples('../../../data/test_data/WNUT/17/wnut17train.conll')
dataset_wnut17_test = get_tuples('../../../data/test_data/WNUT/17/emerging.test.annotated')

dataset_ritters_train = get_tuples('../../../data/test_data/ritter_ner.tsv')

# cnt = 0
# rem = []
#
# for sent1 in dataset_ritters_train:
#     for sent2 in dataset_wnut16_train:
#         sim = 0
#         for tok1 in sent1:
#             for tok2 in sent2:
#                 if tok1 == tok2:
#                     sim+=1
#         if abs(sim - len(sent1)) < 2 and abs(sim - len(sent2)) < 2:
#             cnt+=1
#             rem.append(sent1)
# print cnt, len(dataset_ritters_train)
# for sent1 in dataset_ritters_train:
#     for sent2 in dataset_wnut17_train:
#         sim = 0
#         for tok1 in sent1:
#             for tok2 in sent2:
#                 if tok1 == tok2:
#                     sim+=1
#         if abs(sim - len(sent1)) < 2 and abs(sim - len(sent2)) < 2:
#             cnt+=1
#             rem.append(sent1)
#
# print cnt, len(dataset_ritters_train)
#
# for elem in rem:
#     if elem in dataset_ritters_train:
#         dataset_ritters_train.remove(elem)
# print cnt, len(dataset_ritters_train)

train_sents = dataset_ritters_train
train_sents.extend(dataset_wnut15_train)
train_sents.extend(dataset_wnut17_train)
train_sents.extend(dataset_wnut17_test)


test_sents =  dataset_wnut16_test
test_sents.extend(dataset_wnut16_train)

tf_idf_clone_1 = joblib.load('../../../one-hot-classifiers/tf-idf+svm_1.pkl')
tf_idf_clone_2 = joblib.load('../../../one-hot-classifiers/tf-idf+svm_2.pkl')
tf_idf_clone_3 = joblib.load('../../../one-hot-classifiers/tf-idf+svm_3.pkl')
tf_idf_clone = joblib.load('../../../multi-class-classifier/tf-idf+svm/tf-idf+svm_new.pkl')


#extract new features
print "Extracting new features"
X_train_new = [sent2features_new(s) for s in train_sents]
X_test_new = [sent2features_new(s) for s in test_sents]

#extract raw labels
print "Extracting labels"
y_train_raw = [sent2labels(s) for s in train_sents]
y_test_raw = [sent2labels(s) for s in test_sents]

print "Grouping labels"
#grouping raw labels
y_train = group_labels(y_train_raw)
y_test = group_labels(y_test_raw)

#dumping new features to avoid losing them later
print "Dumping new features"
joblib.dump(X_train_new, 'X_train_new.pkl', compress=9)
joblib.dump(X_test_new, 'X_test_new.pkl', compress=9)

#dump labels
print "Dumping labels"
joblib.dump(y_train, 'y_train.pkl', compress=9)
joblib.dump(y_test, 'y_test.pkl', compress=9)

# load new features initially
print "Loading new features"
X_train_new = joblib.load('X_train_new.pkl')
X_test_new = joblib.load('X_test_new.pkl')
#
# #extract old features from new features
print "Extracting old features"
X_train, X_test = remove_extra_features(X_train_new, X_test_new)

#dump old set of features
print "Dumping old features"
joblib.dump(X_train, 'X_train.pkl', compress=9)
joblib.dump(X_test, 'X_test.pkl', compress=9)

#load new features to restore them to their original state
print "Loading new features to restore them to their original state"
X_train_new = joblib.load('X_train_new.pkl')
X_test_new = joblib.load('X_test_new.pkl')

#load old features
print "Loading old features"
X_train = joblib.load('X_train.pkl')
X_test = joblib.load('X_test.pkl')

#load labels
print "Loading labels"
y_train = joblib.load('y_train.pkl')
y_test = joblib.load('y_test.pkl')

print len(y_train), len(X_train_new)
print len(y_train), len(X_train)



print "start"

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

crf.fit(X_train, y_train)
crf_new.fit(X_train_new, y_train)

joblib.dump(crf, 'crf-suite-old.pkl', compress=9)
joblib.dump(crf_new, 'crf-suite-new.pkl', compress=9)

ner_new = joblib.load('crf-suite-new.pkl')
ner_old = joblib.load('crf-suite-old.pkl')

new_pred = ner_new.predict(X_test_new)
old_pred = ner_old.predict(X_test)

sorted_labels = definitions.KLASSES.copy()
del sorted_labels[4]

print "-----------------------------------------"
print(flat_classification_report(y_test, new_pred, labels=sorted_labels.values(), digits=3,
                                 target_names=sorted_labels.values()))
print(flat_classification_report(y_test, old_pred, labels=sorted_labels.values(), digits=3,
                                 target_names=sorted_labels.values()))
print "-----------------------------------------"
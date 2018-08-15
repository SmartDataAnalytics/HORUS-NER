
import nltk
import sklearn_crfsuite
from sklearn_crfsuite.metrics import flat_classification_report, flat_f1_score, flatten
from nltk import word_tokenize, pos_tag, ne_chunk
from sklearn.externals import joblib
from sklearn.metrics import  f1_score
from	sklearn.model_selection	import	StratifiedKFold, cross_val_predict,KFold, cross_val_score, cross_validate
from sklearn.preprocessing import MultiLabelBinarizer
from	sklearn.base	import	clone
from nltk.chunk import conlltags2tree, tree2conlltags
# from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
import definitions

tf_idf_clone_1 = joblib.load('../one-hot-classifiers/tf-idf+svm_1.pkl')
tf_idf_clone_2 = joblib.load('../one-hot-classifiers/tf-idf+svm_2.pkl')
tf_idf_clone_3 = joblib.load('../one-hot-classifiers/tf-idf+svm_3.pkl')
tf_idf_clone = joblib.load('../multi-class-classifier/tf-idf+svm/tf-idf+svm_new.pkl')


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



def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2features_new(sent):
    return [word2features_new(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label.encode("utf-8") for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

ner_new = joblib.load('crf-suite-new.pkl')
ner_old = joblib.load('crf-suite-old.pkl')

CONST_WIKI_ALL = "../data/test_data/ritter_ner.tsv"

# dataset = np.genfromtxt(CONST_WIKI_ALL, delimiter='\t', skip_header=1)

import csv
X_test_final_old = []
X_test_final_new = []
y_test_final = []


with open(CONST_WIKI_ALL,'rb') as tsvin, open('new.csv', 'wb') as csvout:

    sent = ""
    labels = []
    # try:
    for word in tsvin:
        word = word.split("\t")
        word = [w.replace("\n", "") for w in word]

        if word[0] == '':
            splitted = sent.split(" ")
            splitted = [str.strip(w) for w in splitted]
            # splitted = [re.sub('[^A-Za-z0-9]+', '', w) for w in splitted]
            splitted = [w for w in splitted if len(w) >= 1]
            # print splitted
            X_test_final_old.append(sent2features((tree2conlltags(ne_chunk(pos_tag(splitted))))))
            y_test_final.append(labels)
            sent = ""
            labels = []
        else:
            # if len(word[0].split(" ")) > 1:
            #     print word[0].split(" ")
            if len(word) > 2:
                print word
            sent = sent + " " + str.strip(word[0])
            labels.append(word[1])

with open(CONST_WIKI_ALL,'rb') as tsvin, open('new.csv', 'wb') as csvout:

    sent = ""
    labels = []
    # try:
    for word in tsvin:
        word = word.split("\t")
        word = [w.replace("\n", "") for w in word]

        if word[0] == '':
            splitted = sent.split(" ")
            splitted = [str.strip(w) for w in splitted]
            # splitted = [re.sub('[^A-Za-z0-9]+', '', w) for w in splitted]
            splitted = [w for w in splitted if len(w) >= 1]
            # print splitted
            X_test_final_new.append(sent2features_new((tree2conlltags(ne_chunk(pos_tag(splitted))))))
            # y_test_final.append(labels)
            sent = ""
            labels = []
        else:
            # if len(word[0].split(" ")) > 1:
            #     print word[0].split(" ")
            if len(word) > 2:
                print word
            sent = sent + " " + str.strip(word[0])
            labels.append(word[1])

idx = 0
print y_test_final

print (ner_new.score(X_test_final_new, y_test_final), "new_model")
print (ner_old.score(X_test_final_old, y_test_final), "old_model")

new_pred = ner_new.predict(X_test_final_new)
old_pred = ner_old.predict(X_test_final_old)

#TODO: move this into a method

old = []
new = []
y =[]

for string in old_pred:
    temp = []
    for tok in string:
        if tok.find("LOC") != -1 or tok.find("loc")!=-1:
            temp.append(1)
        else:
            if tok.find("ORG") != -1 or tok.find('org')!=-1 or tok.find('company')!=-1:
                temp.append(2)
            else:
                if tok.find("PER") != -1 or tok.find("per")!=-1 or tok.find("musicartist")!=-1:
                    temp.append(3)
                else:
                    if tok.find("MISC") != -1:
                        temp.append(4)
                    else:
                        temp.append(4)

    old.append(temp)

for string in new_pred:
    temp = []
    for tok in string:
        if tok.find("LOC") != -1 or tok.find("loc") != -1:
            temp.append(1)
        else:
            if tok.find("ORG") != -1 or tok.find('org') != -1 or tok.find('company') != -1:
                temp.append(2)
            else:
                if tok.find("PER") != -1 or tok.find("per") != -1 or tok.find("musicartist") != -1:
                    temp.append(3)
                else:
                    if tok.find("MISC") != -1:
                        temp.append(4)
                    else:
                        temp.append(4)
    new.append(temp)

for string in y_test_final:
    temp = []
    for tok in string:
        if tok.find("LOC") != -1 or tok.find("loc") != -1:
            temp.append(1)
        else:
            if tok.find("ORG") != -1 or tok.find('org')!=-1 or tok.find('company') != -1:
                temp.append(2)
            else:
                if tok.find("PER") != -1 or tok.find("per")!= -1 or tok.find("musicartist") != -1:
                    temp.append(3)
                else:
                    if tok.find("MISC") != -1:
                        temp.append(4)
                    else:
                        temp.append(4)

    y.append(temp)

sorted_labels = definitions.KLASSES.copy()
del sorted_labels[4]

print("------------------------------------------------------")
print flat_f1_score(y, new, average='weighted', labels=sorted_labels.keys())
print flat_f1_score(y, old, average='weighted', labels=sorted_labels.keys())
print "-----------------------------------------"
print(flat_classification_report(y, new, labels=sorted_labels.keys(), target_names=sorted_labels.values(), digits=3))
print(flat_classification_report(y, old, labels=sorted_labels.keys(), target_names=sorted_labels.values(), digits=3))



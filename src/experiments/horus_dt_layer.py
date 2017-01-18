import csv

import itertools

import numpy
from sklearn import tree, preprocessing

# "IS_ENTITY?", "ID_SENT", "ID_WORD", "WORD/TERM", "POS_UNI", "POS", "NER", "COMPOUND", "COMPOUND_SIZE", "ID_TERM_TXT",
# "ID_TERM_IMG", "TOT_IMG", "TOT_CV_LOC", "TOT_CV_ORG", "TOT_CV_PER", "DIST_CV_I", "PL_CV_I", "CV_KLASS",
# "TOT_RESULTS_TX", "TOT_TX_LOC", "TOT_TX_ORG", "TOT_TX_PER", "TOT_ERR_TRANS", "DIST_TX_I", "TX_KLASS", "HORUS_KLASS"
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

ner_ritter_per = ['B-person', 'I-person']
ner_ritter_org = ['B-company', 'I-company']
ner_ritter_loc = ['B-geo-loc', 'I-geo-loc']

file1reader = csv.reader(open("/Users/dnes/Github/components-models/components/out_exp000_0.csv"), delimiter=",")
header1 = file1reader.next()

features = []
X, Y = [], []

for linha in file1reader:
    if linha[6] in ner_ritter_per:
        Y.append(1)
    elif linha[6] in ner_ritter_loc:
        Y.append(2)
    elif linha[6] in ner_ritter_org:
        Y.append(3)
    else:
        Y.append(4)

    features.append((linha[5], int(linha[12]), int(linha[13]), int(linha[14]), int(linha[15]),
                    int(linha[16]), int(linha[19]), int(linha[20]), int(linha[21]),
                    float(linha[22]), int(linha[23])))


features = numpy.array(features)
pos = features[:,0]

le = preprocessing.LabelEncoder()
le.fit(pos)

for x in features:
    x[0] = le.transform(x[0])

#vec = DictVectorizer()
#d = dict(itertools.izip_longest(*[iter(features)] * 2, fillvalue=""))
#X = vec.fit_transform([item[0] for item in d]).toarray()

X_train, X_test, Y_train, Y_test = train_test_split(features, Y, train_size=0.50, test_size=0.50)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)

print len(X_train), len(Y_train)
print len(X_test), len(Y_test)

print '--FI', precision_score(Y_test, predictions, average=None)
print '--FI', recall_score(Y_test, predictions, average=None)
print '--FI', f1_score(Y_test, predictions, average=None)
print '--FI', accuracy_score(Y_test, predictions, normalize=True)

print '--FI', confusion_matrix(Y_test, predictions)


#print Y_test
#print predictions

# list(le.inverse_transform([2, 2, 1]))

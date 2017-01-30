import csv
import pandas
import itertools
import numpy
from sklearn import tree, preprocessing, ensemble, svm
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from horus import definitions
from horus.components.config import HorusConfig

config = HorusConfig()
file1reader = csv.reader(open(config.output_path + "experiments/ritter/EXP_000/out_exp000_1.csv"), delimiter=",")
header1 = file1reader.next()

features = []
X, Y = [], []

colnames = ['IS_ENTITY', 'ID_SENT', 'ID_WORD', 'WORD_TERM', 'POS_UNI',
            'POS', 'NER', 'COMPOUND', 'COMPOUND_SIZE', 'ID_TERM_TXT',
            'ID_TERM_IMG', 'TOT_IMG', 'TOT_CV_LOC', 'TOT_CV_ORG', 'TOT_CV_PER', 'DIST_CV_I',
            'PL_CV_I', 'CV_KLASS', 'TOT_RESULTS_TX', 'TOT_TX_LOC', 'TOT_TX_ORG', 'TOT_TX_PER',
            'TOT_ERR_TRANS', 'DIST_TX_I', 'TX_KLASS', 'HORUS_KLASS']

for linha in file1reader:
    if linha[6] in definitions.NER_RITTER_PER:
        Y.append(3)
    elif linha[6] in definitions.NER_RITTER_LOC:
        Y.append(1)
    elif linha[6] in definitions.NER_RITTER_ORG:
        Y.append(2)
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

X_train, X_test, Y_train, Y_test = train_test_split(features, Y, train_size=0.80, test_size=0.20)

clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best')
model = clf.fit(X_train, Y_train)
predictions = model.predict(X_test)

clf2 = ensemble.RandomForestClassifier(n_estimators=20)
model2 = clf2.fit(X_train, Y_train)
predictions2 = model2.predict(X_test)


clf3 = tree.ExtraTreeClassifier()
model3 = clf3.fit(X_train, Y_train)
predictions3 = model3.predict(X_test)

#clf4 = ensemble.RandomTreesEmbedding()
#model4 = clf4.fit(X_train, Y_train)
#predictions4 = model4.predict(X_test)

clf5 = ensemble.VotingClassifier(estimators=[('lr', clf), ('rf', clf2), ('gnb', clf3)], voting='hard')
model5 = clf5.fit(X_train, Y_train)
predictions5 = model5.predict(X_test)




#print len(X_train), len(Y_train)
#print len(X_test), len(Y_test)

#print '--FI', precision_score(Y_test, predictions, average=None)
#print '--FI', recall_score(Y_test, predictions, average=None)
#print '--FI', f1_score(Y_test, predictions, average=None)
#print '--FI', accuracy_score(Y_test, predictions, normalize=True)


#print '--FI', confusion_matrix(Y_test, predictions)
target_names = ['LOC', 'ORG', 'PER', 'NON']
print(classification_report(Y_test, predictions, target_names=target_names, digits=3))
print(classification_report(Y_test, predictions2, target_names=target_names, digits=3))
print(classification_report(Y_test, predictions3, target_names=target_names, digits=3))
#print(classification_report(Y_test, predictions4, target_names=target_names, digits=3))
print(classification_report(Y_test, predictions5, target_names=target_names, digits=3))

print 'media_mod1',  sum(f1_score(Y_test, predictions, average=None)[0:3])/3.0
print 'media_mod2',  sum(f1_score(Y_test, predictions2, average=None)[0:3])/3.0
print 'media_mod3',  sum(f1_score(Y_test, predictions3, average=None)[0:3])/3.0
#print 'media_mod4',  sum(f1_score(Y_test, predictions4, average=None)[0:3])/3.0
print 'media_mod5',  sum(f1_score(Y_test, predictions5, average=None)[0:3])/3.0



#print Y_test
#print predictions

# list(le.inverse_transform([2, 2, 1]))

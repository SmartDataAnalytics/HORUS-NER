import csv
import re

import pandas
import itertools
import numpy
import sklearn
from sklearn import tree, preprocessing, ensemble, svm
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.svm import SVC

from horus import definitions
from horus.components.config import HorusConfig

config = HorusConfig()
#file1reader = csv.reader(open(config.output_path + "experiments/ritter/EXP_000/out_exp000_1.csv"), delimiter=",")
#header1 = file1reader.next()

features = []
X, Y = [], []

df = pandas.read_csv(config.output_path + "experiments/ritter/EXP_000/out_exp000_1.csv", delimiter=",", skiprows=1, header=None)
for index, linha in df.iterrows():
    pos_bef = ''
    pos_aft = ''
    if index > 1:
        pos_bef = df.get_value(index-1,5)
    if index + 1 < len(df):
        pos_aft = df.get_value(index+1,5)

    if linha[6] in definitions.NER_RITTER_LOC: Y.append(1)
    elif linha[6] in definitions.NER_RITTER_ORG: Y.append(2)
    elif linha[6] in definitions.NER_RITTER_PER: Y.append(3)
    else: Y.append(4)

    one_char_token = 1 if len(linha[3]) == 1 else 0
    special_char = 1 if len(re.findall('(http://\S+|\S*[^\w\s]\S*)',linha[3]))>0 else 0
    first_capitalized = 1 if linha[3][0].isupper() else 0
    capitalized = 1 if linha[3].isupper() else 0
    '''
    pos-1; pos; pos+1; cv_loc; cv_org; cv_per; cv_dist; cv_plc; 
    tx_loc; tx_org; tx_per; tx_err; tx_dist; 
    one_char; special_char; first_cap; cap
    '''
    features.append((pos_bef, linha[5], pos_aft, linha[3], int(linha[12]), int(linha[13]), int(linha[14]), int(linha[15]),
                     int(linha[16]), int(linha[19]), int(linha[20]), int(linha[21]),
                     float(linha[22]), int(linha[23]), one_char_token, special_char, first_capitalized, capitalized))


features = numpy.array(features)


pos = []
pos.extend(features[:,0])
pos.extend(features[:,1])
pos.extend(features[:,2])
pos.extend(features[:,3])

le = preprocessing.LabelEncoder()
le.fit(pos)
__ = joblib.dump(le, 'final_encoder.pkl', compress=3)

for x in features:
    x[0] = le.transform(x[0])
    x[1] = le.transform(x[1])
    x[2] = le.transform(x[2])
    x[3] = le.transform(x[3])

#vec = DictVectorizer()
#d = dict(itertools.izip_longest(*[iter(features)] * 2, fillvalue=""))
#X = vec.fit_transform([item[0] for item in d]).toarray()

X_train, X_test, Y_train, Y_test = train_test_split(features, Y, train_size=0.70, test_size=0.30)

#with open("autiputi2.csv", "w") as output:
#    writer = csv.writer(output, lineterminator='\n')
#    writer.writerows([Y_test])

#output.close()
#exit(0)

#clf10 = sklearn.naive_bayes.
#model10 = clf10.fit(X_train, Y_train)
#predictions10 = model10.predict(X_test)

#clf0 = OneVsRestClassifier(SVC(kernel='linear',class_weight='auto'),n_jobs=-1)
#model0 = clf0.fit(X_train, Y_train)
#predictions0 = model0.predict(X_test)

''' -> quit good recall??
.19", DeprecationWarning)
             precision    recall  f1-score   support

        LOC      0.125     0.452     0.196       104
        ORG      0.036     0.674     0.068        46
        PER      0.172     0.812     0.284       186
        NON      0.998     0.867     0.928     13649

avg / total      0.977     0.863     0.911     13985
'''



clf11 = GaussianNB()
model11 = clf11.fit(np.array(X_train).astype(float), np.array(Y_train).astype(float))
predictions11 = model11.predict(np.array(X_test).astype(float))

clf2 = ensemble.RandomForestClassifier(n_estimators=20)
model2 = clf2.fit(X_train, Y_train)
predictions2 = model2.predict(X_test)

_ = joblib.dump(model2, 'final_randomforest.pkl', compress=3)

clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best')
model = clf.fit(X_train, Y_train)
predictions = model.predict(X_test)

clf3 = tree.ExtraTreeClassifier()
model3 = clf3.fit(X_train, Y_train)
predictions3 = model3.predict(X_test)

clf4 =  tree.DecisionTreeClassifier(criterion='entropy', splitter='best')
model4 = clf4.fit(X_train, Y_train)
predictions4 = model4.predict(X_test)

clf5 = ensemble.VotingClassifier(estimators=[('lr', clf), ('rf', clf2), ('gnb', clf3)], voting='hard')
model5 = clf5.fit(X_train, Y_train)
predictions5 = model5.predict(X_test)

clf6 =  tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_features='auto')
model6 = clf6.fit(X_train, Y_train)
predictions6 = model6.predict(X_test)

clf7 =  tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_features='auto', max_depth=10)
model7 = clf7.fit(X_train, Y_train)
predictions7 = model7.predict(X_test)

clf8 = ensemble.RandomForestClassifier(n_estimators=5)
model8 = clf8.fit(X_train, Y_train)
predictions8 = model8.predict(X_test)

clf9 = ensemble.RandomForestClassifier(n_estimators=50)
model9 = clf9.fit(X_train, Y_train)
predictions9 = model9.predict(X_test)



#print len(X_train), len(Y_train)
#print len(X_test), len(Y_test)

#print '--FI', precision_score(Y_test, predictions, average=None)
#print '--FI', recall_score(Y_test, predictions, average=None)
#print '--FI', f1_score(Y_test, predictions, average=None)
#print '--FI', accuracy_score(Y_test, predictions, normalize=True)


#print '--FI', confusion_matrix(Y_test, predictions)


target_names = ['LOC', 'ORG', 'PER', 'NON']

#print(classification_report(Y_test, predictions0, target_names=target_names, digits=3))
print(classification_report(Y_test, predictions, target_names=target_names, digits=3))
print(classification_report(Y_test, predictions2, target_names=target_names, digits=3))
#print(classification_report(Y_test, predictions10, target_names=target_names, digits=3))
print(classification_report(Y_test, predictions11, target_names=target_names, digits=3))

print(classification_report(Y_test, predictions3, target_names=target_names, digits=3))
print(classification_report(Y_test, predictions4, target_names=target_names, digits=3))
print(classification_report(Y_test, predictions5, target_names=target_names, digits=3))
print(classification_report(Y_test, predictions6, target_names=target_names, digits=3))
print(classification_report(Y_test, predictions7, target_names=target_names, digits=3))
print(classification_report(Y_test, predictions8, target_names=target_names, digits=3))
print(classification_report(Y_test, predictions9, target_names=target_names, digits=3))


#print 'media_mod1',  sum(f1_score(Y_test, predictions0, average=None)[0:3])/3.0
print 'media_mod1',  sum(f1_score(Y_test, predictions, average=None)[0:3])/3.0
print 'media_mod2',  sum(f1_score(Y_test, predictions2, average=None)[0:3])/3.0
#print 'media_mod2',  sum(f1_score(Y_test, predictions10, average=None)[0:3])/3.0
print 'media_mod11',  sum(f1_score(Y_test, predictions11, average=None)[0:3])/3.0
print 'media_mod3',  sum(f1_score(Y_test, predictions3, average=None)[0:3])/3.0
print 'media_mod4',  sum(f1_score(Y_test, predictions4, average=None)[0:3])/3.0
print 'media_mod5',  sum(f1_score(Y_test, predictions5, average=None)[0:3])/3.0
print 'media_mod6',  sum(f1_score(Y_test, predictions6, average=None)[0:3])/3.0
print 'media_mod7',  sum(f1_score(Y_test, predictions7, average=None)[0:3])/3.0
print 'media_mod8',  sum(f1_score(Y_test, predictions8, average=None)[0:3])/3.0
print 'media_mod9',  sum(f1_score(Y_test, predictions9, average=None)[0:3])/3.0

#print Y_test
#print predictions

# list(le.inverse_transform([2, 2, 1]))

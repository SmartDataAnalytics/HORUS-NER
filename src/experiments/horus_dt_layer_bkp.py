import numpy
import pandas
from sklearn import tree, preprocessing, ensemble
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from horus import definitions
from horus.core import HorusConfig

config = HorusConfig()
#file1reader = csv.reader(open(config.output_path + "experiments/ritter/EXP_000_baseline_simple/out_exp000_1_NLTK.csv"), delimiter=",")
#header1 = file1reader.next()

features = []
X, Y = [], []

colnames = ['IS_ENTITY', 'ID_SENT', 'ID_WORD', 'WORD_TERM', 'POS_UNI',
            'POS', 'NER', 'COMPOUND', 'COMPOUND_SIZE', 'ID_TERM_TXT',
            'ID_TERM_IMG', 'TOT_IMG', 'TOT_CV_LOC', 'TOT_CV_ORG', 'TOT_CV_PER', 'DIST_CV_I',
            'PL_CV_I', 'CV_KLASS', 'TOT_RESULTS_TX', 'TOT_TX_LOC', 'TOT_TX_ORG', 'TOT_TX_PER',
            'TOT_ERR_TRANS', 'DIST_TX_I', 'TX_KLASS', 'HORUS_KLASS']

data = pandas.read_csv((config.output_path + "experiments/ritter/EXP_000_baseline_simple/out_exp000_1_NLTK.csv"), sep=',',
                       names=colnames, na_values=['*'], header=0,
                       dtype={"IS_ENTITY?": int, "ID_SENT": int, "ID_WORD": int, "WORD_TERM": str,
                              "POS_UNI": str, "POS": str, "NER": str, "COMPOUND": int, "COMPOUND_SIZE": int,
                              "ID_TERM_TXT": int, "ID_TERM_IMG": int, "TOT_IMG": int, "TOT_CV_LOC": int,
                              "TOT_CV_ORG": int, "TOT_CV_PER": int, "DIST_CV_I": int, "PL_CV_I": int,
                              "CV_KLASS": str, "TOT_RESULTS_TX": int, "TOT_TX_LOC": int, "TOT_TX_ORG": int,
                              "TOT_TX_PER": int, "TOT_ERR_TRANS": int, "DIST_TX_I": int, "TX_KLASS": str,
                              "HORUS_KLASS": str})
_pos = data.POS.tolist()

POS_PREV_3, POS_PREV_2, POS_PREV_1, POS_PLUS_1, POS_PLUS_2, POS_PLUS_3 = '', '', '', '', '', ''

tot = len(data)
pos_array = data.POS.tolist()
for index, row in data.iterrows():
    if index >= 3:
        POS_PREV_3 = pos_array[index - 3]
    if index >= 2:
        POS_PREV_2 = pos_array[index - 2]
    if index >= 1:
        POS_PREV_1 = pos_array[index - 1]
    if (index + 1) < tot:
        POS_PLUS_1 = pos_array[index + 1]
    if (index + 2) < tot:
        POS_PLUS_2 = pos_array[index + 2]
    if (index + 3) < tot:
        POS_PLUS_3 = pos_array[index + 3]

    if row['NER'] in definitions.NER_RITTER_LOC:
        Y.append(1)
    elif row['NER'] in definitions.NER_RITTER_ORG:
        Y.append(2)
    elif row['NER'] in definitions.NER_RITTER_PER:
        Y.append(3)
    else:
        Y.append(4)

    #POS_PREV_3, POS_PREV_2, POS_PREV_1, POS_PLUS_1, POS_PLUS_2, POS_PLUS_3
    #features.append((row['POS'],
    #                 int(row['TOT_CV_LOC']),
    #                 int(row['TOT_CV_ORG']),
    #                 int(row['TOT_CV_PER']),
    #                 int(row['DIST_CV_I']),
    #                 int(row['PL_CV_I'])))

    features.append((row['POS'],
                     #POS_PREV_3,
                     POS_PREV_2,
                     POS_PREV_1,
                     POS_PLUS_1,
                     POS_PLUS_2,
                     #POS_PLUS_3,
                     int(row['TOT_TX_LOC']),
                     int(row['TOT_TX_ORG']),
                     int(row['TOT_TX_PER']),
                     int(row['DIST_TX_I']),
                     int(row['TOT_ERR_TRANS'])))

    #,int(row['TOT_TX_LOC']), int(row['TOT_TX_ORG']), int(row['TOT_TX_PER']),
    #int(row['TOT_ERR_TRANS']), int(row['DIST_TX_I'])
    POS_PREV_3, POS_PREV_2, POS_PREV_1, POS_PLUS_1, POS_PLUS_2, POS_PLUS_3 = '', '', '', '', '', ''

features = numpy.array(features)
pos = features[:,0]
clear_final = numpy.concatenate((pos, ['']))


le = preprocessing.LabelEncoder()
le.fit(clear_final)

for x in features:
    x[0] = le.transform(x[0])
    x[1] = le.transform(x[1])
    x[2] = le.transform(x[2])
    x[3] = le.transform(x[3])
    x[4] = le.transform(x[4])
    #x[5] = le.transform(x[5])
    #x[6] = le.transform(x[6])

#vec = DictVectorizer()
#d = dict(itertools.izip_longest(*[iter(features)] * 2, fillvalue=""))
#X = vec.fit_transform([item[0] for item in d]).toarray()

X_train, X_test, Y_train, Y_test = train_test_split(features, Y, train_size=0.70, test_size=0.30)

clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='best')
model = clf.fit(X_train, Y_train)
predictions = model.predict(X_test)

scores = cross_val_score(clf, features, Y, cv=5, scoring='f1_macro')
print scores

clf2 = ensemble.RandomForestClassifier(n_estimators=16)

scores2 = cross_val_score(clf2, features, Y, cv=5, scoring='f1_macro')
print scores2

model2 = clf2.fit(X_train, Y_train)
predictions2 = model2.predict(X_test)

print len(X_train), len(Y_train)
print len(X_test), len(Y_test)

print '--FI', precision_score(Y_test, predictions, average=None)
print '--FI', recall_score(Y_test, predictions, average=None)
print '--FI', f1_score(Y_test, predictions, average=None)
print '--FI', accuracy_score(Y_test, predictions, normalize=True)

print '--FI', confusion_matrix(Y_test, predictions)
target_names = ['LOC', 'ORG', 'PER', 'NON']
print(classification_report(Y_test, predictions, target_names=target_names, digits=3))
print(classification_report(Y_test, predictions2, target_names=target_names, digits=3))

print 'media_mod1',  sum(f1_score(Y_test, predictions, average=None)[0:3])/3.0
print 'media_mod2',  sum(f1_score(Y_test, predictions2, average=None)[0:3])/3.0

#print Y_test
#print predictions

# list(le.inverse_transform([2, 2, 1]))

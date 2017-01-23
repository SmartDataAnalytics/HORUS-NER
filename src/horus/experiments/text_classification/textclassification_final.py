import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from nltk.stem import *
from numpy import concatenate
from sklearn.ensemble import RandomForestClassifier
import HTMLParser
from sklearn.externals import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from time import time


target_names = ['LOC', 'ORG', 'PER']
MODELS_TEXT_DIR = '/Users/esteves/Github/horus-models/src/horus/resource/models/horus/text/'

CONST_WIKI_ALL = '/Users/esteves/Github/horus-models/data/dataset/Wikipedia/wiki_3classes2.csv'
train_ds = np.genfromtxt(CONST_WIKI_ALL, delimiter="|\-/|", skip_header=1,
                         dtype={'names': ('klass', 'text'), 'formats': (np.int, '|S1000')})

train = np.array(train_ds)
y_train = train['klass']


print("Extracting features from the training data using a sparse vectorizer")
t0 = time()
#vectorizer = TfidfVectorizer( =True, max_df=0.5, stop_words='english',
#                             strip_accents='unicode', encoding='utf-8',decode_error='ignore')

vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                               strip_accents='unicode', encoding='utf-8', decode_error='ignore')


X_train = vectorizer.fit_transform(train['text'])
duration = time() - t0
print("done in %fs" % (duration))


##########################
# TEST DATA
##########################

colnames = ['object', 'abstract']
df1 = pd.read_csv(("/Users/esteves/Github/horus-models/data/dataset/DBPedia/dbo_LOC.csv"), sep=',',
                       names=colnames, header=0, dtype={"object": str, "abstract": str})
df2 = pd.read_csv(("/Users/esteves/Github/horus-models/data/dataset/DBPedia/dbo_ORG.csv"), sep=',',
                       names=colnames, header=0, dtype={"object": str, "abstract": str})
df3 = pd.read_csv(("/Users/esteves/Github/horus-models/data/dataset/DBPedia/dbo_PER.csv"), sep=',',
                       names=colnames, header=0, dtype={"object": str, "abstract": str})
df1['klass'] = 1
df2['klass'] = 2
df3['klass'] = 3

frames = [df1, df2, df3]
dffinal = pd.concat(frames)

y_test = dffinal.klass.tolist()

print("Extracting features from the test data using the same vectorizer")
t0 = time()
X_test = vectorizer.transform(dffinal.abstract.tolist())
duration = time() - t0
print("done in %fs" % (duration))

print("Extracting %d best features by a chi-squared test" % 100)
t0 = time()
ch2 = SelectKBest(chi2, k=100)
X_train = ch2.fit_transform(X_train, y_train)
X_test = ch2.transform(X_test)
print("done in %fs" % (time() - t0))

def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."

def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.f1_score(y_test, pred)
    print("f1:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

    print("classification report:")
    print(metrics.classification_report(y_test, pred, target_names=target_names))

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time

results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(n_estimators=100), "Random forest")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                            dual=False, tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty)))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet")))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))

print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results.append(benchmark(Pipeline([
  ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
  ('classification', LinearSVC())
])))

joblib.dump(results, 'text_classification_results2b.pkl', compress=3)

import pandas as pd
import numpy as np
import random
import string
import sys
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score,f1_score, roc_curve, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

reload(sys)
sys.setdefaultencoding('utf8')

lemma = WordNetLemmatizer()
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
feature_extraction = TfidfVectorizer()
#useful methods
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

def preprocessing(abstract, klass, y_n):
    doc_clean = [clean(doc).split() for doc in abstract]
    cnt = 0

    doc_tfidf = []
    for line in doc_clean:
        tmp = ""
        i = 0
        for word in line:
            tmp = tmp + word
            if i < (len(line) - 1):
                tmp = tmp + " "
        doc_tfidf.append(tmp)

    doc_tfidf_clean = []
    labels_clean = []
    idx = 0
    for word in doc_tfidf:
        if len(word) > 2:
            doc_tfidf_clean.append(word)
            labels_clean.append(y_n[idx])
        idx += 1

    data_iter = []
    aux = 0

    for t in doc_clean:
        data_iter.extend(t)


    X = feature_extraction.fit_transform(doc_tfidf_clean)

    label_final = []

    for label in labels_clean:
        if label == klass:
            label_final.append(True)
        else:
            label_final.append(False)

    return doc_tfidf_clean, X, label_final

#load data
#ps: data_n means preparing data for class n

Locations = pd.read_csv('../data/dbpedia/final_data/Location.csv')
Persons = pd.read_csv('../data/dbpedia/final_data/Person.csv')
Organizations = pd.read_csv('../data/dbpedia/final_data/Organization.csv')
Others = pd.read_csv('../data/dbpedia/final_data/Other.csv')

#TODO: pull other
#prepare data for location model
person_1 = Persons.sample(frac=1).reset_index(drop=True) #shuffle and pick top 33k from each other class
person_1 = person_1.head(33000)

other_1 = Others.sample(frac=1).reset_index(drop=True) #shuffle and pick top 33k from each other class
other_1 = other_1.head(33000)

organization_1 = Organizations.sample(frac=1).reset_index(drop=True) #shuffle and pick top 33k from each other class
organization_1 = organization_1.head(33000)


# dataset_1 = pd.DataFrame(location_1, organizations_1, columns=['class', 'abstract'])
person_1 = person_1.values.tolist()
organization_1 = organization_1.values.tolist()
location_1 = Locations.values.tolist()
other_1 = other_1.values.tolist()
dataset_1 = []

for per in person_1:
    dataset_1.append(per)
for org in organization_1:
    dataset_1.append(org)
for loc in location_1:
    dataset_1.append(loc)
for other in other_1:
    dataset_1.append(other)

random.shuffle(dataset_1)

X_1, y_1 = [], []
for data in dataset_1:
    data[1] = unicode(data[1])
    data[1] = data[1].encode('utf-8')
    y_1.append(data[0])
    X_1.append(data[1])
#remove stop words

doc_tfidf_clean, X_1, y_1 = preprocessing(X_1, 1, y_1)


# X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.2, random_state=5)
#
# clf_1 = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=1000, random_state=42)
# clf_1.fit(X_train_1, y_train_1)
# joblib.dump(clf_1, 'tf-idf+svm_1.pkl', compress=9) # use to save model
#
# predictions = clf_1.predict(X_test_1)
# print('f1', f1_score(y_test_1, predictions))
# print('ROC-AUC yields ' + str(accuracy_score(y_test_1, predictions)))
# print(clf_1.score(X_test_1, y_test_1))

parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),
                  'clf-svm__alpha': (1e-2, 1e-3)}
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42))])
#tune
from sklearn.model_selection import GridSearchCV
gs_clf = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
X_train, X_test, y_train, y_test = train_test_split(doc_tfidf_clean, y_1, test_size=0.2, random_state=5)
gs_clf = gs_clf.fit(X_train, y_train)
final_clf = gs_clf.best_estimator_.fit(X_train, y_train)
joblib.dump(final_clf, 'tf-idf+svm_1.pkl', compress=9) # use to save model
predictions = final_clf.predict(X_test)
print('f1', f1_score(y_test, predictions))
# #####################################################################################################################
# #model 2
#
#
# prepare data for organization model
person_2 = Persons.sample(frac=1).reset_index(drop=True) #shuffle and pick top 33k from each other class
person_2 = person_2.head(33000)

location_2 = Locations.sample(frac=1).reset_index(drop=True) #shuffle and pick top 33k from each other class
location_2 = location_2.head(33000)

other_2 = Others.sample(frac=1).reset_index(drop=True) #shuffle and pick top 33k from each other class
other_2 = other_2.head(33000)

#TODO: do for other

# dataset_1 = pd.DataFrame(location_1, organizations_1, columns=['class', 'abstract'])
person_2 = person_2.values.tolist()
location_2 = location_2.values.tolist()
organization_2 = Organizations.values.tolist()
other_2 = other_2.values.tolist()
dataset_2 = []

for per in person_2:
    dataset_2.append(per)
for org in organization_2:
    dataset_2.append(org)
for loc in location_2:
    dataset_2.append(loc)
for other in other_2:
    dataset_2.append(other)

random.shuffle(dataset_2)

X_2, y_2 = [], []
for data in dataset_2:
    data[1] = unicode(data[1])
    data[1] = data[1].encode('utf-8')
    y_2.append(data[0])
    X_2.append(data[1])

#remove stop words

doc_tfidf_clean, X_2, y_2 = preprocessing(X_2, 2, y_2)


# X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.2, random_state=5)
#
# clf_2 = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=1000, random_state=42)
# clf_2.fit(X_train_2, y_train_2)
# joblib.dump(clf_2, 'tf-idf+svm_2.pkl', compress=9) # use to save model
#
# predictions = clf_2.predict(X_test_2)
# print('f1', f1_score(y_test_2, predictions))
# print('ROC-AUC yields ' + str(accuracy_score(y_test_2, predictions)))
# print(clf_2.score(X_test_2, y_test_2))
parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),
                  'clf-svm__alpha': (1e-2, 1e-3)}
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42))])
#tune
from sklearn.model_selection import GridSearchCV
gs_clf = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
X_train, X_test, y_train, y_test = train_test_split(doc_tfidf_clean, y_2, test_size=0.2, random_state=5)
gs_clf = gs_clf.fit(X_train, y_train)
final_clf = gs_clf.best_estimator_.fit(X_train, y_train)
joblib.dump(final_clf, 'tf-idf+svm_2.pkl', compress=9) # use to save model
predictions = final_clf.predict(X_test)
print('f1', f1_score(y_test, predictions))

print "done with 2"
# #####################################################################################################################
#
#prepare data for organization model

organization_3 = Organizations.sample(frac=1).reset_index(drop=True) #shuffle and pick top 33k from each other class
organization_3 = organization_3.head(33000)

location_3 = Locations.sample(frac=1).reset_index(drop=True) #shuffle and pick top 33k from each other class
location_3 = location_3.head(33000)

other_3 = Others.sample(frac=1).reset_index(drop=True) #shuffle and pick top 33k from each other class
other_3 = other_3.head(33000)

#TODO: do for other

# # dataset_1 = pd.DataFrame(location_1, organizations_1, columns=['class', 'abstract'])
person_3 = Persons.values.tolist()
location_3 = location_3.values.tolist()
organization_3 = organization_3.values.tolist()
other_3 = other_3.values.tolist()
dataset_3 = []

for per in person_3:
    dataset_3.append(per)
for org in organization_3:
    dataset_3.append(org)
for loc in location_3:
    dataset_3.append(loc)
for other in other_3:
    dataset_3.append(other)

random.shuffle(dataset_3)

X_3, y_3 = [], []
for data in dataset_3:

    data[1] = unicode(data[1])
    data[1] = data[1].encode('utf-8')
    y_3.append(data[0])
    X_3.append(data[1])

#remove stop words
doc_tfidf_clean, X_3, y_3 = preprocessing(X_3, 3, y_3)


# X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_3, y_3, test_size=0.2, random_state=5)
#
# clf_3 = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, random_state=42)
# clf_3.fit(X_train_3, y_train_3)


# predictions = clf_3.predict(X_test_3)
# print('f1', f1_score(y_test_3, predictions))
# print('ROC-AUC yields ' + str(accuracy_score(y_test_3, predictions)))
# print(clf_3.score(X_test_3, y_test_3))
# print (X_3)

parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),
                  'clf-svm__alpha': (1e-2, 1e-3)}
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42))])
#tune
from sklearn.model_selection import GridSearchCV
gs_clf = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
X_train, X_test, y_train, y_test = train_test_split(doc_tfidf_clean, y_3, test_size=0.2, random_state=5)
gs_clf = gs_clf.fit(X_train, y_train)
final_clf = gs_clf.best_estimator_.fit(X_train, y_train)
predictions = final_clf.predict(X_test)
print('f1', f1_score(y_test, predictions))
joblib.dump(final_clf, 'tf-idf+svm_3.pkl', compress=9) # use to save model
tf_idf_clone_3 = joblib.load('tf-idf+svm_3.pkl')
print tf_idf_clone_3.predict(["hady is a good boy"])
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

#methods
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


#Load all data

Locations = pd.read_csv('../data/dbpedia/final_data/Location.csv')
Persons = pd.read_csv('../data/dbpedia/final_data/Person.csv')
Organizations = pd.read_csv('../data/dbpedia/final_data/Organization.csv')
Others = pd.read_csv('../data/dbpedia/final_data/Other.csv')

organization_3 = Organizations.sample(frac=1).reset_index(drop=True) #shuffle and pick top 33k from each other class
organization_3 = organization_3.head(100000)

location_3 = Locations.sample(frac=1).reset_index(drop=True) #shuffle and pick top 33k from each other class
location_3 = location_3.head(100000)

other_3 = Others.sample(frac=1).reset_index(drop=True) #shuffle and pick top 33k from each other class
other_3 = other_3.head(100000)


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
doc_tfidf_clean, X_3, y_3 = preprocessing(X_3, 1, y_3)


# X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_3, y_3, test_size=0.2, random_state=5)
#
# clf_3 = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, random_state=42)
# clf_3.fit(X_train_3, y_train_3)


# predictions = clf_3.predict(X_test_3)
# print('f1', f1_score(y_test_3, predictions))
# print('ROC-AUC yields ' + str(accuracy_score(y_test_3, predictions)))
# print(clf_3.score(X_test_3, y_test_3))
# print (X_3)


tf_idf_clone_3 = joblib.load('tf-idf+svm_3.pkl')
tf_idf_clone_2 = joblib.load('tf-idf+svm_2.pkl')
tf_idf_clone_1 = joblib.load('tf-idf+svm_1.pkl')

predictions = tf_idf_clone_1.predict(doc_tfidf_clean)
print('f1_3', f1_score(y_3, predictions))

# doc_tfidf_clean, X_3, y_3 = preprocessing(X_3, 2, y_3)
#
# predictions = tf_idf_clone_2.predict(doc_tfidf_clean)
# print('f1_2', f1_score(y_3, predictions))
#
# doc_tfidf_clean, X_3, y_3 = preprocessing(X_3, 1, y_3)
#
# predictions = tf_idf_clone_1.predict(doc_tfidf_clean)
# print('f1_1', f1_score(y_3, predictions))
#
# print tf_idf_clone_3.predict(["hady is a good boy"])

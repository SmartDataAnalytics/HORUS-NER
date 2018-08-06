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

#preprocessing methods methods
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

def preprocessing(abstract, y_n):
    doc_clean = [clean(doc).split() for doc in abstract]

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

    X = feature_extraction.fit_transform(doc_tfidf_clean)

    return doc_tfidf_clean, X, labels_clean


#Load all data

Locations = pd.read_csv('../../data/dbpedia/final_data/Location.csv')
Persons = pd.read_csv('../../data/dbpedia/final_data/Person.csv')
Organizations = pd.read_csv('../../data/dbpedia/final_data/Organization.csv')
Others = pd.read_csv('../../data/dbpedia/final_data/Other.csv')

#---------------------------------  preprocessing -------------------------------

# organization = Organizations.sample(frac=1).reset_index(drop=True) #shuffle
# organization = organization.head(100000)
#
# location = Locations.sample(frac=1).reset_index(drop=True) #shuffle
# location = location.head(100000)
#
# other = Others.sample(frac=1).reset_index(drop=True) #shuffle
# other = other.head(100000)

person = Persons.values.tolist()
location = Locations.values.tolist()
organization = Organizations.values.tolist()
other = Others.values.tolist()
dataset = []

for per in person:
    dataset.append(per)
for org in organization:
    dataset.append(org)
for loc in location:
    dataset.append(loc)
for other in other:
    dataset.append(other)

random.shuffle(dataset)

X, y = [], []
for data in dataset:

    data[1] = unicode(data[1])
    data[1] = data[1].encode('utf-8')
    y.append(data[0])
    X.append(data[1])

#remove stop words
doc_tfidf_clean, X, y = preprocessing(X, y)

parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),
                  'clf-svm__alpha': (1e-2, 1e-3)}
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42))])
#tune
from sklearn.model_selection import GridSearchCV
gs_clf = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
X_train, X_test, y_train, y_test = train_test_split(doc_tfidf_clean, y, test_size=0.2, random_state=5)
gs_clf = gs_clf.fit(X_train, y_train)
final_clf = gs_clf.best_estimator_.fit(X_train, y_train)
joblib.dump(final_clf, 'tf-idf+svm_new.pkl', compress=9) # use to save model
tf_idf_clone_3 = joblib.load('tf-idf+svm_new.pkl')

print tf_idf_clone_3.predict(["hady is a good boy"])
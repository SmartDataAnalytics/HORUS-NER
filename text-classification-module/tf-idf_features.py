import numpy as np
import nltk
import string
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score,f1_score, roc_curve, confusion_matrix
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split, cross_val_predict
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer


# read data

CONST_WIKI_ALL = "data/wiki_3classes2.csv"

dataset = np.genfromtxt(CONST_WIKI_ALL, delimiter="|\-/|", skip_header=1,
dtype={'names': ('klass', 'text'), 'formats': (np.int, '|S1000')})

docs = dataset['text']
labels = dataset['klass']

lemma = WordNetLemmatizer()

#fix the document
fixed = []

aux = 0
for line in docs:
    line = line.strip()
    line = line.decode('cp1252')
    fixed.extend([nltk.re.sub(r'[^\x00-\x7F]+', ' ', line)])

doc_complete = fixed
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in doc_complete]

doc_tfidf = []
for line in doc_clean:
    tmp = ""
    i = 0
    for word in line:
        tmp = tmp + word
        if i < (len(line) - 1):
            tmp = tmp + " "
    doc_tfidf.append(tmp)

# print(len(doc_tfidf))
doc_tfidf_clean = []
labels_clean = []
idx = 0
for word in doc_tfidf:
    if len(word) > 2:
        doc_tfidf_clean.append(word)
        labels_clean.append(labels[idx])
    idx += 1

data_iter = []
aux = 0
for t in doc_clean:
    data_iter.extend(t)

feature_extraction = TfidfVectorizer()
X = feature_extraction.fit_transform(doc_tfidf_clean)

X_train, X_test, y_train, y_test = train_test_split(X, labels_clean, test_size=0.2, random_state=5)

clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, random_state=42)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

print('ROC-AUC yields ' + str(accuracy_score(y_test, predictions)))
print(clf.score(X_test, y_test))


parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),
                  'clf-svm__alpha': (1e-2, 1e-3)}
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42))])
#tune
from sklearn.model_selection import GridSearchCV
gs_clf = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
X_train, X_test, y_train, y_test = train_test_split(doc_tfidf_clean, labels_clean, test_size=0.2, random_state=5)
gs_clf = gs_clf.fit(X_train, y_train)
print(gs_clf.best_score_)
print(gs_clf.best_params_)
final_clf = gs_clf.best_estimator_.fit(X_train, y_train)
# joblib.dump(final_clf, 'tf-idf+svm.pkl', compress=9) # use to save model
predictions = final_clf.predict(X_test)
print('ROC-AUC yields ' + str(accuracy_score(y_test, predictions)))
print(final_clf.score(X_test, y_test))

y_train_pred = cross_val_predict(gs_clf,	X_train,	y_train,	cv=3)
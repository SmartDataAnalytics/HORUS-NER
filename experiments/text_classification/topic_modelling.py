from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
import nltk
import numpy as np
from collections import Counter
from itertools import chain
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split

#https://de.dariah.eu/tatom/topic_model_mallet.html#topic-model-mallet
#https://de.dariah.eu/tatom/topic_model_python.html
#https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730
#https://medium.com/towards-data-science/improving-the-interpretation-of-topic-models-87fd2ee3847d
#http://miriamposner.com/blog/very-basic-strategies-for-interpreting-results-from-the-topic-modeling-tool/
#https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/
from horus.core.config import HorusConfig

config = HorusConfig()

'''
POS filter -> remove IN, CD, MD
Frequency filter -> remove all w which have count(w)<x
Batch Wise LDA -> create batches of fixed sizes, run LDA multiple times on these, get intersection of all.
'''
def get_histogram(sentences):
    #words_notflat = [a.split() for a in sentences]
    counter = Counter(chain.from_iterable(sentences))
    # print counter

#doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
#doc2 = "My father spends a lot of time driving my sister around to dance practice."
#doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
#doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
#doc5 = "Health experts say that Sugar is not good for your lifestyle."

#CONST_WIKI_ALL = config.dataset_path + '/Wikipedia/wiki_3classes2.csv'
CONST_WIKI_ALL = config.dataset_path + '/Wikipedia/wiki_4classes.csv'
dataset = np.genfromtxt(CONST_WIKI_ALL, delimiter="|\-/|", skip_header=1,
                         dtype={'names': ('klass', 'text'), 'formats': (np.int, '|S1000')})

dataset = np.array(dataset)
docs = dataset['text']
labels = dataset['klass']

#fixed = [line.decode('utf-8').strip() for line in y_train]
fixed = []

aux = 0
for line in docs:
    # if aux==10000:
    #     break
    line = line.strip()
    #''.join([i if ord(i) < 128 else ' ' for i in line])
    #line = line.decode('utf-8', 'ignore').encode("utf-8")
    fixed.extend([nltk.re.sub(r'[^\x00-\x7F]+', ' ', line)])
    aux+=1


# print str(len(fixed))


#uniarr = []
#uniarr.extend([sentence.encode('ascii', 'ignore') for sentence in y_train])


# compile documents
#doc_complete = [doc1, doc2, doc3, doc4, doc5]
doc_complete = fixed
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in doc_complete]

# get_histogram(doc_clean)
# exit(0)

# Creating the term dictionary of our courpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
corpus = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
# Lda = gensim.models.ldamodel.LdaModel
#
# # Running and Trainign LDA model on the document term matrix.
# ldamodel = Lda(doc_term_matrix, num_topics=10, id2word = dictionary, passes=50)
# # print(ldamodel.print_topics(num_topics=3, num_words=20))
#
# doc_lda_matrix = [ldamodel[doc] for doc in doc_term_matrix]

numpy_matrix = gensim.matutils.corpus2dense(corpus, num_terms=len(dictionary)).T
X_train, X_test, y_train, y_test = train_test_split(np.array(numpy_matrix), labels, test_size=0.2, random_state=5)
print type(X_train), type(y_train)
svm = LinearSVC().fit(X_train, y_train)
print "Accuracy:", svm.score(X_test, y_test)

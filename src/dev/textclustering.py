# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import HTMLParser
from sklearn.externals import joblib
#from bs4 import BeautifulSoup
#import Stemmer



'''
1 = LOC
2 = ORG
3 = PER
'''
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
#N = 10
#a = np.random.rand(N,N)
#b = np.zeros((N,N+1))
#b[:,:-1] = a

from src.config import HorusConfig
config = HorusConfig()

#CONST_WIKI_PER_FILE = config.dir_datasets + '/Wikipedia/wiki_PER2.csv'
#CONST_WIKI_LOC_FILE = config.dir_datasets + '/Wikipedia/wiki_LOC2.csv'
#CONST_WIKI_ORG_FILE = config.dir_datasets + '/Wikipedia/wiki_ORG2.csv'
CONST_WIKI_ALL = config.dir_datasets + '/Wikipedia/wiki_3classes2.csv'


#train_ds = pd.read_csv(CONST_WIKI_PER_FILE, header=0, quoting=3) #engine='python' sep='\t',
#train_ds_loc = np.genfromtxt(CONST_WIKI_LOC_FILE, delimiter="|\-/|", skip_header=1, dtype={'names': ('klass', 'text'), 'formats': (np.int, '|S1000')})
#train_ds_org = np.genfromtxt(CONST_WIKI_ORG_FILE, delimiter="|\-/|", skip_header=1, dtype={'names': ('klass', 'text'), 'formats': (np.int, '|S1000')})
#train_ds_per = np.genfromtxt(CONST_WIKI_PER_FILE, delimiter="|\-/|", skip_header=1, dtype={'names': ('klass', 'text'), 'formats': (np.int, '|S1000')})
train_ds = np.genfromtxt(CONST_WIKI_ALL, delimiter="|\-/|", skip_header=1, dtype={'names': ('klass', 'text'), 'formats': (np.int, '|S1000')})


html_parser = HTMLParser.HTMLParser()
train_ds_clean = html_parser.unescape(train_ds)
print train_ds_clean.shape


Xtrain = np.array(train_ds)

#english_stemmer = Stemmer.Stemmer('en')
#Xtrain = concatenate((train_ds_loc, train_ds_org, train_ds_per),axis=0)
#print ds


#X = np.empty(shape=(len(train_ds), 1)) #numpy.zeros or numpy.ones or numpy.empty
#X = []
#print X.shape
#for example in ds:
#    #temp = example[0].split("|-|")
#    #X.append(([temp[0]], [temp[1]]))
#    #np.append(X, [temp[0], temp[1]], axis=0)
#    X.append((example[0], example[1]))


#loc_data = [x + [1] for x in loc_data]
#for row in loc_data:
#    row.append(1)



''' count words '''

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

if 1==2:
    #count_vect = CountVectorizer(stop_words='english', lowercase=True, strip_accents='unicode', encoding='utf-8',
    #                             decode_error='ignore')

    #X_train_counts = count_vect.fit_transform(Xtrain['text'])
    #tfidf_transformer = TfidfTransformer()
    #X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, strip_accents='unicode', encoding='utf-8',
                                 decode_error='ignore')
    nb = BernoulliNB()
    model = Pipeline([('vectorizer', vectorizer), ('nb', nb)])
    clf = model.fit(Xtrain['text'], Xtrain['klass'])
    _ = joblib.dump(clf, 'nbpipelinetfidf.pkl', compress=3)


docs_per2 = [['A Última Vez Que Vi Richard ilustração. quadrinhos Diego Esteves . Portfólio', 3],
             ['Diego Esteves’ berufliches Profil anzeigen LinkedIn ist das weltweit größte berufliche Netzwerk, das Fach- und Führungskräften wie Diego Esteves ... Diego Esteves | LinkedIn', 3],
             ['Diego Esteves. Systems Engineer at Dell. Location Austin, Texas Area Industry Information Technology and Services Diego Esteves | LinkedIn', 3],
             ['The latest Tweets from Diego Esteves (@estevesdiego): "La Chiqui nada dijo hasta ahora del contrato de $1,5 millones de su productor general y nieto, Nacho ... Diego Esteves (@estevesdiego) | Twitter', 3],
             ['View the profiles of professionals named Diego Esteves on LinkedIn. There are 57 professionals named Diego Esteves, who use LinkedIn to exchange ... Top 10 Diego Esteves profiles | LinkedIn', 3],
             ['The latest Tweets from Diego Esteves (@DiegoEsteves_). Editor de vídeo, Videomaker e Fotografo em Rossini s Imagens http://t.co/nRf00hc4o5 apaixonado por ... Diego Esteves (@DiegoEsteves_) | Twitter', 3],
             ['Diego Esteves is a PhD Student at the University of Leipzig. Diego’s research interests are in the area of Fact Validation on the Web. Research Interests Diego Esteves – Smart Data Analytics - sda.cs.uni-bonn.de', 3],
             ['Email: diegoesteves3d@gmail.com Phone: +55 011-96971-1500 Brazil, Santo André - SP Diego Esteves', 3],
             ['Diego Esteves is on Facebook. Join Facebook to connect with Diego Esteves and others you may know. Facebook gives people the power to share and makes the... Diego Esteves | Facebook', 3],
             ['Ähnliche XING-Profile wie das von DIEGO ESTEVES ALMANZA. Martin Reyes Rico Gerente Comercial Mexico adrian soriano carrillo ... DIEGO ESTEVES ALMANZA - GERENTE - xing.com', 3]]

clf = joblib.load(config.models_text_root + 'nbpipelinetfidf.pkl')
print clf.predict('A Última Vez Que Vi Richard ilustração. quadrinhos Diego Esteves')
#exit(0)



count_vect = CountVectorizer(stop_words='english', lowercase=True, strip_accents='unicode',
                             encoding='utf-8', decode_error='ignore')
X_train_counts = count_vect.fit_transform(Xtrain['text'])
print X_train_counts.shape
joblib.dump(X_train_counts, config.models_text_root + 'xtrain-counts.data', compress=3)
#print count_vect.vocabulary_.get(u'algorithm')

''' tf-idf '''


#tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
#X_train_tf = tf_transformer.transform(X_train_counts)
#print X_train_tf.shape
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print X_train_tfidf.shape
joblib.dump(X_train_counts, config.models_text_root + 'xtrain-counts-tfidf.data', compress=3)

''' feature_extraction '''


clf = MultinomialNB().fit(X_train_tfidf, np.array(Xtrain['klass'], dtype=np.int32))
joblib.dump(clf, config.models_text_root + 'multiNB.pkl', compress=3)

clf2 = BernoulliNB().fit(X_train_tfidf, np.array(Xtrain['klass'], dtype=np.int32))
joblib.dump(clf2, config.models_text_root + 'bernoulliNB.pkl', compress=3)

clf3 = SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet").fit(X_train_tfidf, np.array(Xtrain['klass'], dtype=np.int32))
joblib.dump(clf3, config.models_text_root + 'sgd.pkl', compress=3)

clf4 = LinearSVC(loss='l2', dual=False, tol=1e-3).fit(X_train_tfidf, np.array(Xtrain['klass'], dtype=np.int32))
joblib.dump(clf4, config.models_text_root + 'linearSVCloss.pkl', compress=3)

clf5 = MultinomialNB(alpha=.01).fit(X_train_tfidf, np.array(Xtrain['klass'], dtype=np.int32))
joblib.dump(clf5, config.models_text_root + 'multiNBa01.pkl', compress=3)

clf6 = BernoulliNB(alpha=.01).fit(X_train_tfidf, np.array(Xtrain['klass'], dtype=np.int32))
joblib.dump(clf6, config.models_text_root + 'bernoulliNBa01.pkl', compress=3)

clf7 = NearestCentroid().fit(X_train_tfidf, np.array(Xtrain['klass'], dtype=np.int32))
joblib.dump(clf7, config.models_text_root + 'NC.pkl', compress=3)

clf8 = LinearSVC().fit(X_train_tfidf, np.array(Xtrain['klass'], dtype=np.int32))
joblib.dump(clf8, config.models_text_root + 'linearSVC.pkl', compress=3)



''' testing DBPedia '''
colnames = ['object', 'abstract']
df1 = pd.read_csv((config.dir_datasets + "/DBPedia/dbo_LOC.csv"), sep=',',
                       names=colnames, header=0,
                       dtype={"object": str, "abstract": str})
df2 = pd.read_csv((config.dir_datasets + "/DBPedia/dbo_ORG.csv"), sep=',',
                       names=colnames, header=0,
                       dtype={"object": str, "abstract": str})
df3 = pd.read_csv((config.dir_datasets + "/DBPedia/dbo_PER.csv"), sep=',',
                       names=colnames, header=0,
                       dtype={"object": str, "abstract": str})
#loc_data = data.abstract.tolist()

df1['klass'] = 1
df2['klass'] = 2
df3['klass'] = 3

frames = [df1, df2, df3]
dffinal = pd.concat(frames)

Y_test = dffinal.klass.tolist()

X_new_counts = count_vect.transform(dffinal.abstract.tolist())
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)
predicted2 = clf2.predict(X_new_tfidf)
predicted3 = clf3.predict(X_new_tfidf)
predicted4 = clf4.predict(X_new_tfidf)
predicted5 = clf5.predict(X_new_tfidf)
predicted6 = clf6.predict(X_new_tfidf)
predicted7 = clf7.predict(X_new_tfidf)
predicted8 = clf8.predict(X_new_tfidf)


target_names = ['LOC', 'ORG', 'PER']
print(classification_report(Y_test, predicted, target_names=target_names, digits=3))
print(classification_report(Y_test, predicted2, target_names=target_names, digits=3))
print(classification_report(Y_test, predicted3, target_names=target_names, digits=3))
print(classification_report(Y_test, predicted4, target_names=target_names, digits=3))
print(classification_report(Y_test, predicted5, target_names=target_names, digits=3))
print(classification_report(Y_test, predicted6, target_names=target_names, digits=3))
print(classification_report(Y_test, predicted7, target_names=target_names, digits=3))
print(classification_report(Y_test, predicted8, target_names=target_names, digits=3))








''' testing '''

docs_loc = [['Belo Horizonte is the capital of the state of Minas Gerais, Brazil s second most populous state. ... The city features a mixture of contemporary and classical buildings, and is home to several modern Brazilian architectural icons', 1],
            ['Angola is a country in Southern Africa. It is the seventh-largest country in Africa, and is bordered by Namibia to the', 1],
            ['officially the Macao Special Administrative Region of the People s Republic of China, is an autonomous territory on', 1],
            ['The Amazon rainforest also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South', 1],
            ['New Zealand is an island nation in the southwestern Pacific Ocean. The country geographically ', 1],
            ['Japan is an island country in East Asia. Located in the Pacific Ocean, it lies to the east of the ... Japan is pronounced Nippon or Nihon. The English word Japan possibly derives from the historical', 1]]


docs_org = [['Bayern Munich is a German sports club based in Munich, Bavaria, Germany. It is best known for its professional football team, which plays in the Bundesliga ...', 2],
            ['Microsoft Corporation is an American multinational technology company headquartered in Redmond, Washington, that ...', 2],
            ['Burger King (often abbreviated as BK) is an American global chain of hamburger fast food restaurants, headquartered in the unincorporated area of Miami-Dade ...', 2],
            ['Starbucks Corporation is an American coffee company and coffeehouse chain. Starbucks was founded in Seattle, Washington in 1971. Today it operates 23,768 ...', 2],
            ['SC Victoria Hamburg is a German association football club from the city of Hamburg. ... It was one of the founding members of the DFB', 2],
            ['A strip mall is an open-air shopping mall where the stores are arranged in a row, with a sidewalk in front. Strip malls are typically developed as a unit and have...', 2]]

docs_per = [['Michael Hoffman, Director: One Fine Day. Michael Hoffman was born on November 30, 1956 in Honolulu, Hawaii, USA. He is a director and writer, known for ...', 3],
            ['By now you must have seen Michael Hoffman, all of Michael Hoffman, every nook and cranny of Michael Hoffman. The bodybuilder, tumblr star, ', 3],
            ['Michael Lynn Hoffman (born November 30, 1956) is an American film director. Contents. [hide]. 1 Early life and education; 2 Career; 3 Personal life ...', 3],
            ['Revisionist historian Michael Hoffman "A scrupulously liberating desire for truth-telling." Independent History & Research Box 849  Coeur d', 3],
            ['Remember Michael Hoffman? He is the tattooed beefcake who made ripples across the web when a video of him pleasuring the muscle ...', 3],
            [' Here is a transcript of Hoffman s apology video: Fuuuccckkk! Hey everyone, it is Michael. It is been quite a while since I made a video', 3]]


docs_per2 = [['A Última Vez Que Vi Richard ilustração. quadrinhos Diego Esteves . Portfólio', 3],
             ['Diego Esteves’ berufliches Profil anzeigen LinkedIn ist das weltweit größte berufliche Netzwerk, das Fach- und Führungskräften wie Diego Esteves ... Diego Esteves | LinkedIn', 3],
             ['Diego Esteves. Systems Engineer at Dell. Location Austin, Texas Area Industry Information Technology and Services Diego Esteves | LinkedIn', 3],
             ['The latest Tweets from Diego Esteves (@estevesdiego): "La Chiqui nada dijo hasta ahora del contrato de $1,5 millones de su productor general y nieto, Nacho ... Diego Esteves (@estevesdiego) | Twitter', 3],
             ['View the profiles of professionals named Diego Esteves on LinkedIn. There are 57 professionals named Diego Esteves, who use LinkedIn to exchange ... Top 10 Diego Esteves profiles | LinkedIn', 3],
             ['The latest Tweets from Diego Esteves (@DiegoEsteves_). Editor de vídeo, Videomaker e Fotografo em Rossini s Imagens http://t.co/nRf00hc4o5 apaixonado por ... Diego Esteves (@DiegoEsteves_) | Twitter', 3],
             ['Diego Esteves is a PhD Student at the University of Leipzig. Diego’s research interests are in the area of Fact Validation on the Web. Research Interests Diego Esteves – Smart Data Analytics - sda.cs.uni-bonn.de', 3],
             ['Email: diegoesteves3d@gmail.com Phone: +55 011-96971-1500 Brazil, Santo André - SP Diego Esteves', 3],
             ['Diego Esteves is on Facebook. Join Facebook to connect with Diego Esteves and others you may know. Facebook gives people the power to share and makes the... Diego Esteves | Facebook', 3],
             ['Ähnliche XING-Profile wie das von DIEGO ESTEVES ALMANZA. Martin Reyes Rico Gerente Comercial Mexico adrian soriano carrillo ... DIEGO ESTEVES ALMANZA - GERENTE - xing.com', 3]]

docs_new = np.append(docs_loc, docs_org, axis=0)
docs_new = np.append(docs_new, docs_per, axis=0)
docs_new = np.append(docs_new, docs_per2, axis=0)

X_new_counts = count_vect.transform(docs_new[:, 0])
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)
predicted2 = clf2.predict(X_new_tfidf)
predicted3 = clf3.predict(X_new_tfidf)
predicted4 = clf4.predict(X_new_tfidf)
predicted5 = clf5.predict(X_new_tfidf)
predicted6 = clf6.predict(X_new_tfidf)
predicted7 = clf7.predict(X_new_tfidf)
predicted8 = clf8.predict(X_new_tfidf)

hit1, hit2, hit3, hit4, hit5, hit6, hit7, hit8 = 0, 0, 0, 0, 0, 0, 0, 0
tot = len(docs_new)

print '*********************************'
print 'predictions model 1'
for doc, real, category in zip(docs_new[:, 0], docs_new[:][:,1], predicted):
    #print('%r = %s => %s' % (doc[:20], real, category))
    if int(real) == int(category):
        hit1 = hit1 + 1
print 'accuracy is %s' % str(float(hit1) / tot)

print '*********************************'
print 'predictions model 2'
for doc, real, category in zip(docs_new[:, 0], docs_new[:][:,1], predicted2):
    #print('%r = %s => %s' % (doc[:20], real, category))
    if int(real) == int(category):
        hit2 = hit2 + 1
print 'accuracy is %s' % str(float(hit2) / tot)

print '*********************************'
print 'predictions model 3'
for doc, real, category in zip(docs_new[:, 0], docs_new[:][:,1], predicted3):
    #print('%r = %s => %s' % (doc[:20], real, category))
    if int(real) == int(category):
        hit3 = hit3 + 1
print 'accuracy is %s' % str(float(hit3) / tot)

print '*********************************'
print 'predictions model 4'
for doc, real, category in zip(docs_new[:, 0], docs_new[:][:,1], predicted4):
    #print('%r = %s => %s' % (doc[:20], real, category))
    if int(real) == int(category):
        hit4 = hit4 + 1
print 'accuracy is %s' % str(float(hit4) / tot)

print '*********************************'
print 'predictions model 5'
for doc, real, category in zip(docs_new[:, 0], docs_new[:][:,1], predicted5):
    #print('%r = %s => %s' % (doc[:20], real, category))
    if int(real) == int(category):
        hit5 = hit5 + 1
print 'accuracy is %s' % str(float(hit5) / tot)

print '*********************************'
print 'predictions model 6'
for doc, real, category in zip(docs_new[:, 0], docs_new[:][:,1], predicted6):
    #print('%r = %s => %s' % (doc[:20], real, category))
    if int(real) == int(category):
        hit6 = hit6 + 1
print 'accuracy is %s' % str(float(hit6) / tot)

print '*********************************'
print 'predictions model 7'
for doc, real, category in zip(docs_new[:, 0], docs_new[:][:,1], predicted7):
    #print('%r = %s => %s' % (doc[:20], real, category))
    if int(real) == int(category):
        hit7 = hit7 + 1
print 'accuracy is %s' % str(float(hit7) / tot)

print '*********************************'
print 'predictions model 8'
for doc, real, category in zip(docs_new[:, 0], docs_new[:][:,1], predicted8):
    #print('%r = %s => %s' % (doc[:20], real, category))
    if int(real) == int(category):
        hit8 = hit8 + 1
print 'accuracy is %s' % str(float(hit8) / tot)


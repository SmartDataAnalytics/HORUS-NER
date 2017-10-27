import re

import numpy
import numpy as np
import pandas
import pydotplus
from IPython.display import Image
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn import tree, ensemble
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB

from horus.core import definitions
from horus.core.config import HorusConfig

config = HorusConfig()
#file1reader = csv.reader(open(config.output_path + "experiments/ritter/EXP_000_baseline_simple/out_exp000_1.csv"), delimiter=",")
#header1 = file1reader.next()


def model_dt(X_train, X_test, y_train, y_test):

    #excluding id_sentence e id_word
    X_train = np.delete(np.array(X_train).astype(float), np.s_[0:2], axis=1)
    X_test = np.delete(np.array(X_test).astype(float), np.s_[0:2], axis=1)

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

    tree.export_graphviz(clf, out_file = 'tree.dot')
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("tree.pdf")

    iris = load_iris()
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=iris.feature_names,
                                    class_names=iris.target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    Image(graph.create_png())
    # http://scikit-learn.org/stable/auto_examples/tree/plot_iris.html#sphx-glr-auto-examples-tree-plot-iris-py
    #print Y_test
    #print predictions

    # list(le.inverse_transform([2, 2, 1]))

def model_lstm(X_train, X_test, y_train, y_test):
    lengths = [len(x) for x in X_train]
    maxlen = max(lengths)
    print('min sentence / max sentence: ', maxlen, min(lengths))


    TOP_WORDS = 500
    MAX_REVIEW_LEN = 50
    X_train = sequence.pad_sequences(X_train, dtype='float')
    X_test = sequence.pad_sequences(X_test)
    print('x_train shape:', X_train.shape)
    print('x_test shape:', X_test.shape)

    print('Build model...')
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(TOP_WORDS, embedding_vecor_length, input_length=MAX_REVIEW_LEN))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    print('train...')
    # model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
    model.fit(X_train, y_train, nb_epoch=3, batch_size=64)
    # Final evaluation of the model

    scores, acc = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    print('Test score:', scores)
    print('Test accuracy:', acc)


features = []
X, Y = [], []
teste = []

samplefile = config.output_path + "experiments/EXP_do_tokenization/out_exp003_wnut16_en_tweetNLP.csv"
df = pandas.read_csv(samplefile, delimiter=",", skiprows=1, header=None, keep_default_na=False, na_values=['_|_'])
for index, linha in df.iterrows():
    if len(linha)>0:
        if linha[7] == 0:

            idsent = linha[1]
            idtoken = idsent = linha[2]

            pos_bef = ''
            pos_aft = ''
            if index > 0 and df.get_value(index-1,7) == 0:
                pos_bef = df.get_value(index-1,5)
            if index + 1 < len(df) and df.get_value(index+1,7) == 0:
                pos_aft = df.get_value(index+1,5)

            if linha[51] in definitions.NER_TAGS_LOC: Y.append(1)
            elif linha[51] in definitions.NER_TAGS_ORG: Y.append(2)
            elif linha[51] in definitions.NER_TAGS_PER: Y.append(3)
            else: Y.append(4)

            token = linha[3]
            pos = linha[5]
            one_char_token = 1 if len(token) == 1 else 0
            special_char = 1 if len(re.findall('(http://\S+|\S*[^\w\s]\S*)', token))>0 else 0
            first_capitalized = 1 if token[0].isupper() else 0
            capitalized = 1 if token.isupper() else 0
            title = 1 if token.istitle() else 0
            digit = 1 if token.isdigit() else 0
            nr_images_returned = linha[17]
            nr_websites_returned = linha[25]
            hyphen = 1 if '-' in token else 0
            cv_loc = int(linha[12])
            cv_org = int(linha[13])
            cv_per = int(linha[14])
            cv_dist = int(linha[15])
            cv_plc = int(linha[16])
            tx_loc = int(linha[20])
            tx_org = int(linha[21])
            tx_per = int(linha[22])
            tx_err = float(linha[23])
            tx_dist = float(linha[24])

            teste.append(linha[6])
            if linha[6] in definitions.NER_TAGS_LOC: ner = definitions.KLASSES2["LOC"]
            elif linha[6] in definitions.NER_TAGS_ORG: ner = definitions.KLASSES2["ORG"]
            elif linha[6] in definitions.NER_TAGS_PER: ner = definitions.KLASSES2["PER"]
            else: ner = definitions.KLASSES2["O"]

            features.append((idsent, idtoken, pos_bef, pos, pos_aft, title, digit,
                             one_char_token, special_char, first_capitalized, hyphen,
                             capitalized, nr_images_returned,
                             cv_org, cv_loc, cv_per, cv_dist, cv_plc,
                             tx_org, tx_loc, tx_per, tx_dist, tx_err))
print len(Y)
print set(Y)
print set(teste)

features = numpy.array(features)


#pos = []
#pos.extend(features[:,0])
#pos.extend(features[:,1])
#pos.extend(features[:,2])
#pos.extend(features[:,3])
#le = preprocessing.LabelEncoder()
#le.fit(pos)
#__ = joblib.dump(le, 'final_encoder.pkl', compress=3)


#if config.models_pos_tag_lib == 1:
#    le = joblib.load(config.encoder_path + "encoder_nltk.pkl")
#elif config.models_pos_tag_lib == 2:
#    le = joblib.load(config.encoder_path + "encoder_stanford.pkl")
#elif config.models_pos_tag_lib == 3:
#    le = joblib.load(config.encoder_path + "encoder_tweetnlp.pkl")

le = joblib.load(config.encoder_path + "_encoder_pos.pkl")
for x in features:
    x[2] = le.transform(x[2])
    x[3] = le.transform(x[3])
    x[4] = le.transform(x[4])

config = HorusConfig()
#features_file = open(config.output_path + 'features_dt_wnut16.csv', 'wb')
#wr = csv.writer(features_file) #quoting=csv.QUOTE_ALL
#wr.writerows(features)

#target_file = open(config.output_path + 'features_dt_wnut16_Y.csv', 'wb')
#wr2 = csv.writer(target_file,  delimiter=";")
#for row in Y:
#    wr2.writerow([row])

#vec = DictVectorizer()
#d = dict(itertools.izip_longest(*[iter(features)] * 2, fillvalue=""))
#X = vec.fit_transform([item[0] for item in d]).toarray()

X_train, X_test, Y_train, Y_test = train_test_split(features, Y, train_size=0.80, test_size=0.20)

model_dt(X_train, X_test, Y_train, Y_test)
#model_lstm(X_train, X_test, Y_train, Y_test)
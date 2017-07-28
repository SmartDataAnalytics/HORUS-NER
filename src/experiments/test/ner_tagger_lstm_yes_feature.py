# https://gist.github.com/zaxtax/9fe6d1809ea3449f29a049b01a3547ec
import nltk
import numpy as np
from keras.engine import InputLayer
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout, Merge
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from keras.utils import to_categorical

# that is a skeleton of how to train NER using LSTM with just a word as
# input feature
raw = open('/Users/esteves/Github/horus-models/data/dataset/Wikiner/wikigold.conll.txt', 'r').readlines()

all_x = []
point = []

for line in raw:
    stripped_line = line.strip().split(' ')
    stripped_line.append(2 if stripped_line[0].isdigit() else 1)
    stripped_line.append(2 if stripped_line[0].isupper() else 1)
    stripped_line.append(2 if stripped_line[0].islower() else 1)
    point.append(stripped_line)
    if line == '\n':
        all_x.append(point[:-1])
        point = []
all_x = all_x[:-1]

short_x = [x for x in all_x if len(x) < 64]

# sentence tokenized
X = [[[c[0], c[2], c[3], c[4]] for c in x] for x in short_x]
# ners
y = [[c[1] for c in y] for y in short_x]

# vector of tokens
all_text = [c[0] for x in X for c in x]

words = list(set(all_text)) # distinct tokens
word2ind = {word: index for index, word in enumerate(words)} # indexes of words
ind2word = {index: word for index, word in enumerate(words)}
labels = list(set([c for x in y for c in x]))
label2ind = {label: (index + 1) for index, label in enumerate(labels)}
ind2label = {(index + 1): label for index, label in enumerate(labels)}
print('Vocabulary size:', len(word2ind), len(label2ind))


lengths = [len(x) for x in X]
maxlen = max(lengths)
print('min sentence / max sentence: ', min(lengths), maxlen)

#y = np.to_categorical(y)

def encode(x, n):
    result = np.zeros(n)
    result[x] = 1
    return result

X_enc = [[[word2ind[c[0]], c[1], c[2], c[3]] for c in x] for x in X]
#X_enc = [[[c[1], c[2]] for c in x] for x in X]
max_label = max(label2ind.values()) + 1
y_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y]
y_enc = [[encode(c, max_label) for c in ey] for ey in y_enc]

X_enc = pad_sequences(X_enc, maxlen=maxlen)
y_enc = pad_sequences(y_enc, maxlen=maxlen)

X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, test_size=0.30, train_size=0.70, random_state=42) #352|1440
print('Training and testing tensor shapes:', X_train.shape, X_test.shape, y_train.shape, y_test.shape)

max_features = len(word2ind)
embedding_size = 128
hidden_size = 128 #32
out_size = len(label2ind) + 1

maxlen = 0
for x in X_train:
    if len(x) > maxlen:
        maxlen = len(x)
print 'max sent: ', maxlen


model1 = Sequential()
model1.add(Embedding(input_dim=max_features, output_dim=embedding_size, input_length=maxlen, mask_zero=True))

model2 = Sequential()
#model2.add(InputLayer(input_dim=3, output_dim=embedding_size, input_length=maxlen, mask_zero=True))
model2.add(InputLayer(input_shape=(maxlen,3)))

model = Sequential()
model.add(Merge([model1, model2], mode = 'concat'))
model.add(Dense(1))
#model.add(Dense(1, init = 'normal', activation = 'sigmoid'))

#model.add(Embedding(input_dim=max_features, output_dim=embedding_size, input_length=maxlen, mask_zero=True))
model.add(LSTM(hidden_size, return_sequences=True, input_shape=(maxlen,3)))
model.add(TimeDistributed(Dense(out_size)))
model.add(Activation('softmax'))
print 'compile...'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
print('train...')


#y_train = np.asarray(y_train).astype(int).reshape(len(y_train), -1)
#print(y_train.shape)
#y_test = np.asarray(y_test).astype(int).reshape(len(y_test), -1)
#print(y_test.shape)

batch_size = 128 #128


model.fit([X_train[:,:,0], X_train[:,:,1:4]], y_train, epochs=17, batch_size=batch_size) #validation_data=(X_test, y_test)
#model.fit(X_train, y_train, epochs=30, batch_size=batch_size) #validation_data=(X_test, y_test)
score = model.evaluate([X_test[:,:,0], X_test[:,:,1:4]], y_test, batch_size=batch_size)
#score = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Raw test score:', score)

def score(yh, pr):
    coords = [np.where(yhh > 0)[0][0] for yhh in yh]
    yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    ypr = [prr[co:] for prr, co in zip(pr, coords)]
    fyh = [c for row in yh for c in row]
    fpr = [c for row in ypr for c in row]
    return fyh, fpr

pr = model.predict_classes([X_train[:,:,0], X_train[:,:,1:4]])
#pr = model.predict_classes(X_train)
yh = y_train.argmax(2)
fyh, fpr = score(yh, pr)
print()
print('Training accuracy:', accuracy_score(fyh, fpr))
print('Training confusion matrix:')
print(confusion_matrix(fyh, fpr))
print precision_recall_fscore_support(fyh, fpr)

pr = model.predict_classes([X_test[:,:,0], X_test[:,:,1:4]])
#pr = model.predict_classes(X_test)
yh = y_test.argmax(2)
fyh, fpr = score(yh, pr)
print('Testing accuracy:', accuracy_score(fyh, fpr))
print('Testing confusion matrix:')
print(confusion_matrix(fyh, fpr))
print precision_recall_fscore_support(fyh, fpr)
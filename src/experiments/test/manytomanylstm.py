import keras.layers as L
import keras.models as M
import numpy
import numpy as np

# The inputs to the model.
# We will create two data points, just for the example.
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

data_x = numpy.array([
    # Datapoint 1
    [
        # Input features at timestep 1
        [1, 2, 3],
        # Input features at timestep 2
        [4, 5, 6]
    ],
    # Datapoint 2
    [
        # Features at timestep 1
        [7, 8, 9],
        # Features at timestep 2
        [10, 11, 12]
    ]
])

# The desired model outputs.
# We will create two data points, just for the example.
data_y = numpy.array([
    # Datapoint 1
    [
        # Target features at timestep 1
        [1, 1, 1],
        # Target features at timestep 2
        [1, 2, 1]
    ],
    # Datapoint 2
    [
        # Target features at timestep 1
        [1, 2, 1],
        # Target features at timestep 2
        [1, 2, 2]
    ]
])


raw = open('/Users/esteves/Github/horus-models/data/dataset/Wikiner/wikigold.conll.txt', 'r').readlines()

all_x = []
point = []

for line in raw:
    stripped_line = line.strip().split(' ')
    stripped_line.append(2 if stripped_line[0].isdigit() else 1)
    stripped_line.append(2 if stripped_line[0].isupper() else 1)
    point.append(stripped_line)
    if line == '\n':
        all_x.append(point[:-1])
        point = []
all_x = all_x[:-1]

short_x = [x for x in all_x if len(x) < 64]


# sentence tokenized
X = [[[c[2],c[3]] for c in x] for x in short_x]
# ners
y = [[c[1] for c in y] for y in short_x]
lengths = [len(x) for x in X]
maxlen = max(lengths)
print('min sentence / max sentence: ', min(lengths), maxlen)

def encode(x, n):
    result = np.zeros(n)
    result[x] = 1
    return result

labels = list(set([c for x in y for c in x]))
label2ind = {label: (index + 1) for index, label in enumerate(labels)}
max_label = max(label2ind.values()) + 1
y_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y]
y_pad_one_hot = [[encode(c, max_label) for c in ey] for ey in y_enc]
#y_pad_one_hot = np.array([to_categorical(sent_label, num_classes=len(labels)+1) for sent_label in y_enc])

X_enc = pad_sequences(X, maxlen=maxlen)
y_enc = pad_sequences(y_pad_one_hot, maxlen=maxlen)

# Each input data point has 2 timesteps, each with 3 features.
# So the input shape (excluding batch_size) is (2, 3), which
# matches the shape of each data point in data_x above.
model_input = L.Input(shape=(maxlen, 2))

# This RNN will return timesteps with 4 features each.
# Because return_sequences=True, it will output 2 timesteps, each
# with 4 features. So the output shape (excluding batch size) is
# (2, 4), which matches the shape of each data point in data_y above.
model_output = L.LSTM(6, return_sequences=True)(model_input)

# Create the model.
model = M.Model(input=model_input, output=model_output)
#model = M.Sequential()
#model.add(L.LSTM(128, return_sequences=True, input_shape=(maxlen, 2)))
#model.add(L.TimeDistributed(L.Dense(6)))
#model.add(L.Activation('softmax'))


# You need to pick appropriate loss/optimizers for your problem.
# I'm just using these to make the example compile.
#model.compile('sgd', 'mean_squared_error', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print model.summary()

X_train, X_test, y_train, y_test = train_test_split(X_enc, y_pad_one_hot, test_size=0.30, train_size=0.70, random_state=42) #352|1440
print('Training and testing tensor shapes:', X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model.fit(X_train, y_train, epochs=17, batch_size=256) #validation_data=(X_test, y_test)
score = model.evaluate(X_test, y_test)

print('Raw test score:', score)


def score(yh, pr):
    coords = [np.where(yhh > 0)[0][0] for yhh in yh]
    yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    ypr = [prr[co:] for prr, co in zip(pr, coords)]
    fyh = [c for row in yh for c in row]
    fpr = [c for row in ypr for c in row]
    return fyh, fpr

#pr = model.predict_classes(X_train)
pr = model.predict(X_train)
yh = y_train.argmax(2)
fyh, fpr = score(yh, pr)
print()
print('Training accuracy:', accuracy_score(fyh, fpr))
print('Training confusion matrix:')
print(confusion_matrix(fyh, fpr))
print precision_recall_fscore_support(fyh, fpr)

pr = model.predict_classes(X_test)
yh = y_test.argmax(2)
fyh, fpr = score(yh, pr)
print('Testing accuracy:', accuracy_score(fyh, fpr))
print('Testing confusion matrix:')
print(confusion_matrix(fyh, fpr))
print precision_recall_fscore_support(fyh, fpr)
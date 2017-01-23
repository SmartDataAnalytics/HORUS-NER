import string

import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib

r = joblib.load('text_classification_results.pkl')

results = []
i = 0
code = {'SGDClassifier': 'SGD', 'RandomForestClassifier': 'Random Forest',
        'KNeighboarsClassifier': 'K-Neighboars', 'PassiveAggressiveClassifier': 'Passive Aggressive',
        'RidgeClassifier': 'Ridge'}
#trans = string.maketrans(*["".join(x) for x in zip(*code.items())])

for temp in r:
    if i not in (13, 9, 5):
        clas = temp[0].replace('Classifier', '')
        results.append([clas, temp[1], temp[2], temp[3]])
    i+=1

indices = np.arange(len(results))
results = [[x[i] for x in results] for i in range(4)]
clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8), facecolor="white")
plt.title("Text Classification: Test Performance (DBPedia - 30K abstracts)", fontweight="bold")
plt.barh(indices, score, .4, label="F-measure", color="gray", linestyle='dashed')
#plt.barh(indices + .3, training_time, .2, label="Training Time")
plt.barh(indices + .6, test_time, .4, label="Test Time", color="red", edgecolor="red")
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.2, i, c, color='black')

for i, c in zip(indices, score):
    plt.text(c+.02, i, "%.2f" % round(c, 2), color='black')


#plt.show()
plt.savefig("textclass.png")
exit(0)
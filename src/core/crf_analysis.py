# feature_extraction
import scipy

import sklearn_crfsuite
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn import metrics as skmetrics

crf2 = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.18687907015736968,
    c2=0.025503200544851036,
    max_iterations=100,
    all_possible_transitions=True
)
# crf2.fit(X_train_CRF_shape, y_train_CRF_shape)

# eval

# y_pred2 = crf2.predict(X_test_CRF_shape)

# metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)

# labels = list(crf.classes_)
# trick for report visualization

# labels = list(['LOC', 'ORG', 'PER'])
# labels.remove('O')
# group B and I results
# sorted_labels = sorted(
#    labels,
#    key=lambda name: (name[1:], name[0])
# )



# print(metrics.flat_classification_report(
#    y_test, y_pred2, labels=sorted_labels, digits=3
# ))
exit(0)
# r = [42, 39, 10, 5, 50]
# fmeasures = []
# for d in range(len(r)):
#    cv_X_train, cv_X_test, cv_y_train, cv_y_test = train_test_split(X_train_CRF_shape, y_train_CRF_shape,
#                                                        test_size = 0.30, random_state = r[d])
#    m = crf.fit(cv_X_train, cv_y_train)
#    cv_y_pred = m.predict(cv_X_test)
#    print(metrics.flat_classification_report(
#        cv_y_test, cv_y_pred, labels=sorted_labels, digits=3
#    ))
# cv_y_test_bin = MultiLabelBinarizer().fit_transform(cv_y_test)
# cv_y_pred_bin = MultiLabelBinarizer().fit_transform(cv_y_pred)
# fmeasures.append(f1_score(cv_y_test_bin, cv_y_pred_bin, average='weighted'))

# print sum(fmeasures)/len(r)

# scores = cross_val_score(crf, _X, _y, cv=5, scoring='f1_macro')
# scores2 = cross_val_score(crf2, _X, _y, cv=5, scoring='f1_macro')

# rs = ShuffleSplit(n_splits=3, test_size=.20, random_state=0)
# for train_index, test_index in rs.split(_X):
#    print("TRAIN:", train_index, "TEST:", test_index)

# print scores
# print scores2

# exit(0)

# define fixed parameters and parameters to search
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=300,
    all_possible_transitions=True
)
params_space = {
    'c1': scipy.stats.expon(scale=0.5),
    'c2': scipy.stats.expon(scale=0.05),
}

# use the same metric for evaluation
f1_scorer = make_scorer(metrics.flat_f1_score,
                        average='weighted', labels=labels)

# search
rs = RandomizedSearchCV(crf, params_space,
                        cv=3,
                        verbose=1,
                        n_jobs=-1,
                        n_iter=50,
                        scoring=f1_scorer)
rs.fit(X_train_CRF_shape, y_train_CRF_shape)

# crf = rs.best_estimator_
print('best params:', rs.best_params_)
print('best CV score:', rs.best_score_)
print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

_x = [s.parameters['c1'] for s in rs.grid_scores_]
_y = [s.parameters['c2'] for s in rs.grid_scores_]
_c = [s.mean_validation_score for s in rs.grid_scores_]

fig = plt.figure()
fig.set_size_inches(12, 12)
ax = plt.gca()
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('C1')
ax.set_ylabel('C2')
ax.set_title("Randomized Hyperparameter Search CV Results (min={:0.3}, max={:0.3})".format(
    min(_c), max(_c)
))

ax.scatter(_x, _y, c=_c, s=60, alpha=0.9, edgecolors=[0, 0, 0])
fig.savefig('crf_optimization.png')

print("Dark blue => {:0.4}, dark red => {:0.4}".format(min(_c), max(_c)))

crf = rs.best_estimator_
y_pred = crf.predict(X_test_CRF_shape)
print(metrics.flat_classification_report(
    y_test, y_pred, labels=labels, digits=3
))

from collections import Counter

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

print("Top likely transitions:")
print_transitions(Counter(crf.transition_features_).most_common(20))

print("\nTop unlikely transitions:")
print_transitions(Counter(crf.transition_features_).most_common()[-20:])

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))

print("Top positive:")
print_state_features(Counter(crf.state_features_).most_common(30))

print("\nTop negative:")
print_state_features(Counter(crf.state_features_).most_common()[-30:])

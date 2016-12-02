import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
import logging
import csv
import pandas as pd
import numpy as np
from horus.core import HorusCore
from sklearn.metrics import confusion_matrix


def confusion_matrix_analysis(horus_matrix):
    df = pd.DataFrame(horus_matrix)
    res = df.loc[df[4].isin(ner_ritter)]

    y = res[4]
    y_true = ['PER' if x in ner_ritter_per else 'LOC' if x in ner_ritter_loc else 'ORG' for x in y]
    y_pred_cv = res[13]
    y_pred_tx = res[20]
    cm1 = confusion_matrix(y_true, y_pred_cv)
    cm2 = confusion_matrix(y_true, y_pred_tx)

    plt.figure()
    plot_confusion_matrix(cm1, classes=['PER', 'LOC', 'ORG'], normalize=True,
                          title='CV - Normalized Confusion Matrix')
    plt.show()

    plot_confusion_matrix(cm2, classes=['PER', 'LOC', 'ORG'], normalize=True,
                          title='TX - Normalized Confusion Matrix')
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def example_analysis(horus_matrix):
    tot_loc, tot_org, tot_per, tot_others = 0, 0, 0, 0
    hit_loc, hit_org, hit_per = 0, 0, 0
    hit_loc_tx, hit_org_tx, hit_per_tx = 0, 0, 0
    hit_loc_final, hit_org_final, hit_per_final = 0, 0, 0
    for item in horus_matrix:
        if item[0] == 1:
            #  this indicator -12 plays an important rule
            #if item[13] == 'LOC' and item[12] < 0:
            #    # get the 2nd best candidate
            #    if item[9] >= item[10]:
            #        item[13] = 'ORG'
            #    else:
            #        item[13] = 'PER'
            if item[4] in ner_ritter_per:
                tot_per += 1
                if item[13] == 'PER':
                    hit_per += 1
                if item[20] == 'PER':
                    hit_per_tx += 1
                if item[21] == 'PER':
                    hit_per_final += 1
            elif item[4] in ner_ritter_loc:
                tot_loc += 1
                if item[13] == 'LOC':
                    hit_loc += 1
                if item[20] == 'LOC':
                    hit_loc_tx += 1
                if item[21] == 'LOC':
                    hit_loc_final += 1
            elif item[4] in ner_ritter_org:
                tot_org += 1
                if item[13] == 'ORG':
                    hit_org += 1
                if item[20] == 'ORG':
                    hit_org_tx += 1
                if item[21] == 'ORG':
                    hit_org_final += 1
        else:
            tot_others += 1

    # F1 = 2 * (precision * recall) / (precision + recall)

    logging.info(':: LOC -> hits-cv: %s | %2f | hits-tx: %s | %2f | hits-final: %s | %2f || %s' % (str(hit_loc),
                                                                            float(hit_loc) / tot_loc,
                                                                            str(hit_loc_tx),
                                                                            float(hit_loc_tx) / tot_loc,
                                                                            str(hit_loc_final),
                                                                            float(hit_loc_final) / tot_loc,
                                                                            str(tot_loc)))
    logging.info(':: ORG -> hits-cv: %s | %2f | hits-tx: %s | %2f | hits-final: %s | %2f || %s' % (str(hit_org),
                                                                            float(hit_org) / tot_org,
                                                                            str(hit_org_tx),
                                                                            float(hit_org_tx) / tot_org,
                                                                            str(hit_org_final),
                                                                            float(hit_org_final) / tot_org,
                                                                            str(tot_org)))
    logging.info(':: PER -> hits-cv: %s | %2f | hits-tx: %s | %2f | hits-final: %s | %2f || %s' % (str(hit_per),
                                                                            float(hit_per) / tot_per,
                                                                            str(hit_per_tx),
                                                                            float(hit_per_tx) / tot_per,
                                                                            str(hit_per_final),
                                                                            float(hit_per_final) / tot_per,
                                                                            str(tot_per)))

    logging.info('------------------------------------------------------------------')
    logging.info(':: statistics')
    logging.info(':: number of tokens: ' + str(len(horus_matrix)))


horus = HorusCore('horus.ini')
ner_ritter_per = ['B-person', 'I-person']
ner_ritter_org = ['B-company', 'I-company']
ner_ritter_loc = ['B-geo-loc', 'I-geo-loc']
ner_ritter = []
ner_ritter.extend(ner_ritter_per)
ner_ritter.extend(ner_ritter_org)
ner_ritter.extend(ner_ritter_loc)

with open(horus.horus_final_data_json) as json_data:
    horus_matrix = json.load(json_data)

list1 = [0 ,2]
list2=[["abc", 1, "def"], ["ghi", 2, "wxy"]]
newList = [[each_list[i] for i in list1] for each_list in list2]
# word_term, Y, klass_cv, klass_txt, final_klass
horus_light = [[each_list[i] for i in [3,4,13,20,21]] for each_list in horus_matrix]

hmetalight = open('/Users/dnes/Dropbox/Doutorado_Alemanha/#Papers/#DeFacto Files/horus/cache/horus_out_light.csv', 'wb')
wr = csv.writer(hmetalight, quoting=csv.QUOTE_ALL)
wr.writerows(horus_light)

#with open('/Users/dnes/Dropbox/Doutorado_Alemanha/#Papers/#DeFacto Files/horus/cache/horus_out_light.json', 'wb') as outfile1:
#    json.dump(horus_light, outfile1)

#with open(horus.horus_final_data_csv, 'rb') as f:
#    reader = csv.reader(f)
#    # next(reader) # hack to ignore the header
#    horus_matrix = list(reader)


example_analysis(horus_matrix)

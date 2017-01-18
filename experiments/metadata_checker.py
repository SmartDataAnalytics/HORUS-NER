import csv
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import definitions


file1reader = csv.reader(open(definitions.OUTPUT_PATH + "/horus_out_ritter.csv"), delimiter=",")
header1 = file1reader.next() #header

tot = 0
tper, torg, tloc = 0, 0, 0

tp_per_cv, tp_org_cv, tp_loc_cv = 0, 0, 0
fp_per_cv, fp_org_cv, fp_loc_cv = 0, 0, 0

tp_per_tx, tp_org_tx, tp_loc_tx = 0, 0, 0
fp_per_tx, fp_org_tx, fp_loc_tx = 0, 0, 0

tp_per, tp_org, tp_loc = 0, 0, 0
fp_per, fp_org, fp_loc = 0, 0, 0

y = []
ycv, ytx, yf = [],[],[]
for linha in file1reader:

    NER = linha[6]
    CV_KLASS = linha[17]
    TX_KLASS = linha[24]
    HORUS_KLASS = linha[25]

    if CV_KLASS == 'PER':
        ycv.append(1)
    elif CV_KLASS == 'LOC':
        ycv.append(2)
    elif CV_KLASS == 'ORG':
        ycv.append(3)
    else:
        ycv.append(4)

    if TX_KLASS == 'PER':
        ytx.append(1)
    elif TX_KLASS == 'LOC':
        ytx.append(2)
    elif TX_KLASS == 'ORG':
        ytx.append(3)
    else:
        ytx.append(4)

    if HORUS_KLASS == 'PER':
        yf.append(1)
    elif HORUS_KLASS == 'LOC':
        yf.append(2)
    elif HORUS_KLASS == 'ORG':
        yf.append(3)
    else:
        yf.append(4)

    if NER in definitions.NER_RITTER_PER:
        y.append(1)
    elif NER in definitions.NER_RITTER_LOC:
        y.append(2)
    elif NER in definitions.NER_RITTER_ORG:
        y.append(3)
    else:
        y.append(4)

    tot+=1
    if NER in definitions.NER_RITTER:
        if NER in definitions.NER_RITTER_PER:
            tper+=1
            if CV_KLASS == 'PER':
                tp_per_cv +=1
            else:
                fp_per_cv +=1
            if TX_KLASS == 'PER':
                tp_per_tx +=1
            else:
                fp_per_tx +=1
            if HORUS_KLASS == 'PER':
                tp_per +=1
            else:
                fp_per +=1
        if NER in definitions.NER_RITTER_ORG:
            torg += 1
            if CV_KLASS == 'ORG':
                tp_org_cv +=1
            else:
                fp_org_cv +=1
            if TX_KLASS == 'ORG':
                tp_org_tx +=1
            else:
                fp_org_tx +=1
            if HORUS_KLASS == 'ORG':
                tp_org +=1
            else:
                fp_org +=1
        if NER in definitions.NER_RITTER_LOC:
            tloc += 1
            if CV_KLASS == 'LOC':
                tp_loc_cv +=1
            else:
                fp_loc_cv +=1
            if TX_KLASS == 'LOC':
                tp_loc_tx +=1
            else:
                fp_loc_tx +=1
            if HORUS_KLASS == 'LOC':
                tp_loc +=1
            else:
                fp_loc +=1
    else:
        if CV_KLASS == 'PER':
            fp_per_cv +=1
        if TX_KLASS == 'PER':
            fp_per_tx +=1
        if HORUS_KLASS == 'PER':
            fp_per +=1
        if CV_KLASS == 'ORG':
            fp_org_cv +=1
        if TX_KLASS == 'ORG':
            fp_org_tx +=1
        if HORUS_KLASS == 'ORG':
            fp_org +=1
        if CV_KLASS == 'LOC':
            fp_loc_cv +=1
        if TX_KLASS == 'LOC':
            fp_loc_tx +=1
        if HORUS_KLASS == 'LOC':
            fp_loc +=1

tot_others = tot - (tper+torg+tloc)
print tot, tper, torg, tloc
print tp_per_cv, tp_per_cv/float(tper), tp_org_cv, tp_org_cv/float(torg), tp_loc_cv, tp_loc_cv/float(tloc)
print tp_per_tx, tp_per_tx/float(tper), tp_org_tx, tp_org_tx/float(torg), tp_loc_tx, tp_loc_tx/float(tloc)
print tp_per, tp_per/float(tper), tp_org, tp_org/float(torg), tp_loc, tp_loc/float(tloc)
print '--'
print fp_per_cv, fp_per_cv/float(tot_others), fp_org_cv, fp_org_cv/float(tot_others), fp_loc_cv, fp_loc_cv/float(tot_others)
print fp_per_tx, fp_per_tx/float(tot_others), fp_org_tx, fp_org_tx/float(tot_others), fp_loc_tx, fp_loc_tx/float(tot_others)
print fp_per, fp_per/float(tot_others), fp_org, fp_org/float(tot_others), fp_loc, fp_loc/float(tot_others)

#print ':: Precision (PER, ORG, LOC)'
#print '-- CV', tp_per_cv / float(tp_per_cv + fp_per_cv), \
#      tp_org_cv / float(tp_org_cv + fp_org_cv), \
#      tp_loc_cv / float(tp_loc_cv + fp_loc_cv)
#print '-- TX', tp_per_tx / float(tp_per_tx + fp_per_tx), \
#      tp_org_tx / float(tp_org_tx + fp_org_tx), \
#      tp_loc_tx / float(tp_loc_tx + fp_loc_tx)
#print '-- FI', tp_per / float(tp_per + fp_per), \
#      tp_org / float(tp_org + fp_org), \
#      tp_loc / float(tp_loc + fp_loc)
#print ':: Recall (PER, ORG, LOC)'
#print '-- CV', tp_per_cv / float(tp_per_cv + fn_per_cv), \
#      tp_org_cv / float(tp_org_cv + fp_org_cv), \
#      tp_loc_cv / float(tp_loc_cv + fp_loc_cv)
#print '-- TX', tp_per_tx / float(tp_per_tx + fp_per_tx), \
#      tp_org_tx / float(tp_org_tx + fp_org_tx), \
#      tp_loc_tx / float(tp_loc_tx + fp_loc_tx)
#print '-- FI', tp_per / float(tp_per + fp_per), \
#      tp_org / float(tp_org + fp_org), \
#      tp_loc / float(tp_loc + fp_loc)

print ':: Precision (PER, ORG, LOC, OTHERS)'
print '--CV', precision_score(y, ycv, average=None)
print '--TX', precision_score(y, ytx, average=None)
print '--FI', precision_score(y, yf, average=None)

print ':: Recall (PER, ORG, LOC, OTHERS)'
print '--CV', recall_score(y, ycv, average=None)
print '--TX', recall_score(y, ytx, average=None)
print '--FI', recall_score(y, yf, average=None)

print ':: F-measure (PER, ORG, LOC, OTHERS)'
print '--CV', f1_score(y, ycv, average=None)
print '--TX', f1_score(y, ytx, average=None)
print '--FI', f1_score(y, yf, average=None)

print ':: Accuracy (PER, ORG, LOC)'
print '--CV', accuracy_score(y, ycv, normalize=True)
print '--TX', accuracy_score(y, ytx, normalize=True)
print '--FI', accuracy_score(y, yf, normalize=True)

print ':: Confusion Matrix'
print '--CV', confusion_matrix(y, ycv)
print '--TX', confusion_matrix(y, ytx)
print '--FI', confusion_matrix(y, yf)


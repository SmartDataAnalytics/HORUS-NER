import pandas as pd

import sys
#/home/esteves/github/horus-ner/data/output/EXP_005/metadata_crf_1_to_30.txt
from matplotlib import ticker

metadata = 'NAACL/metadata_crf_1_41_OK.txt'
#metadata = 'NAACL/metadata_trees_1_41_OK.txt'
df=pd.read_table(metadata, sep="\t")

'''
cross-validation chart
'''
df = df[df['algo'] == 'CRF']
df = df[df['task'] == 'NER']
df = df[df['label'] == 'average']
df = df[df['cross-validation'] == True]
#df = df[df['dataset1'] == '2015.conll.freebase.horus']
#df = df[df['dataset2'] == '2016.conll.freebase.ascii.txt.horus']

dictionary = {'ritter.train': 'Ritter',
              'wnut15.train': 'WNUT-15',
              'wnut16.train': 'WNUT-16',
              'wnut17.train': 'WNUT-17'}

datasets = list(set(df['dataset1']))
labels = list(set(df['label']))
configs = list(set(df['config']))
runs = list(set(df['run']))


file = open("process_crf.txt","w")
file.write('dataset'+'\t'+'config'+'\t'+'label'+'\t'+'precision'+'\t'+'recall'+'\t'+'f1'+'\t'+'support'+'\n')

assert datasets == list(set(dictionary.keys()))
#for i in datasets:
for key, value in dictionary.iteritems():
    cur = df[df['dataset1']==key]
    for j in configs:
        conf = cur[cur['config']==j]
        p = 0.0
        r = 0.0
        f = 0.0
        s = 0.0
        for k in runs:
            run = conf[conf['run']==k]
            p += float(run['precision'])
            r += float(run['recall'])
            f += float(run['f1'])
            s += int(run['support'])
        p/=len(runs)
        r/=len(runs)
        f/=len(runs)
        s/=len(runs)
        file.write(str(key)+'\t'+str(j)+'\t'+'avg_'+str(len(runs))+'_fold'+'\t'+str(p)+'\t'+str(r)+'\t'+str(f)+'\t'+str(s))  #replace with write to file
        file.write('\n')

file.close()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

print(len(labels))
print(labels)
gs = gridspec.GridSpec(4, 1) #len(datasets)


df=pd.read_table('process_crf.txt',sep="\t")

fsz = (12, 9)

fig = plt.figure(figsize=fsz)
#fig.autolayout = True

#for i in range(len(dictionary)):
i=0
line_col = [[0,0], [1,0], [2,0], [3,0]]
for key, value in sorted(dictionary.iteritems(), key=lambda (k,v): (v,k)):
    print(key)
    cur = df[df['dataset']==key]
    ax = plt.subplot(gs[line_col[i][0], line_col[i][1]]) #gs[i,0]

    #ax.xaxis.set_xlabel = 'sadasdasdas' #
    plt.xlabel(value, fontsize=11) #, rotation=90, position=(1.03, 50))
    plt.xticks(np.arange(1, len(configs)+1, step=1), fontsize=12)
    plt.yticks(np.arange(0.3, 1.0, step=0.02), fontsize=12)
    p = []
    r = []
    f = []
    top_5 = []

    dict_conf_f1 = {}
    for j in configs:
        con = cur[cur['config'] == j]
        dict_conf_f1[j] = con['f1'].mean()
        f.append(con['f1'].mean())

    dict_conf_f1 = sorted(dict_conf_f1.iteritems(), key=lambda (k, v): (v, k))
    print(dict_conf_f1[0:10])
    print(dict_conf_f1[-5:])



    avg = np.mean(f)
    top_10_worse = np.array(f)
    top_5 = np.array(f)

    top_10_worse = np.sort(top_10_worse)[0:10]
    max_top_10_worse = np.max(top_10_worse)

    top_5 = -np.sort(-top_5)[0:5]
    min_top_5 = np.min(top_5)

    print(np.max(top_5))

    f = []
    for j in configs:
        con = cur[cur['config']==j]
        y = []
        #p.append(con['precision'].mean())
        #r.append(con['recall'].mean())
        f.append(con['f1'].mean())
        #y.append(con['precision'].mean())
        #y.append(con['recall'].mean())
        y.append(con['f1'].mean())

        s = 7**2
        if con['f1'].mean() <= max_top_10_worse:
            c = 'red'
            m = 'x'
        elif con['f1'].mean() >= min_top_5:
            c = 'darkgreen'
            m = '*'
            s = 9**2
        elif con['f1'].mean() >= avg:
            c = 'gray'
            m = "o"
        else:
            c = 'orange'
            m = "o"
        plt.scatter(np.array([j]*1),np.array(y),marker = m, color=c, s=s) #*3 = prec, recall, f1

    ##ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    #ax.plot(np.array(configs),np.array(p),linestyle = '--',color = 'black',label = 'precision')
    #ax.plot(np.array(configs),np.array(r),linestyle = ':',color = 'black',label = 'recall')
    ax.plot(np.array(configs),np.array(f),linestyle = '-',color = 'black', label = 'f1')
    ax.grid(True)
    #ax.xaxis.set_xlabel('dsaads')
    #ax.legend()
    fig.add_subplot(ax)

    i+=1
#plt.show()
plt.tight_layout()
plt.savefig('crf_3fold.png', figsize=fsz, dpi=600, bbox_inches = 'tight')
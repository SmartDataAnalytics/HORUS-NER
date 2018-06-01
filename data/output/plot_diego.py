import pandas as pd

import sys

df=pd.read_table('metadata.txt',sep="\t")

df = df[df['algo'] == 'CRF']

df = df[df['task'] == 'NER']

df = df[df['cross-validation'] == True]

dictionary = {'2015.conll.freebase.horus':'WNUT-15','2016.conll.freebase.ascii.txt.horus':'WNUT-16','emerging.test.annotated.horus':'WNUT-17','ner.txt.horus':'Ritter'}

datasets = list(set(df['dataset1']))
labels = list(set(df['label']))
configs = list(set(df['config']))

file = open("process_crf.txt","w") 

file.write('dataset'+'\t'+'config'+'\t'+'label'+'\t'+'precision'+'\t'+'recall'+'\t'+'f1'+'\t'+'support'+'\n')

for i in datasets:
    cur = df[df['dataset1']==i]
    for j in configs:
        conf = cur[cur['config']==j]
        for k in labels:
            lab = conf[conf['label']==k]
            p = 0.0
            r = 0.0
            f = 0.0
            s = 0.0
            for l in range(lab.shape[0]):
                p+=lab['precision'].iloc[l]
                r+=lab['recall'].iloc[l]
                f+=lab['f1'].iloc[l]
                s+=lab['support'].iloc[l]
            p/=lab.shape[0]
            r/=lab.shape[0]
            f/=lab.shape[0]
            s/=lab.shape[0]
            file.write(str(i)+'\t'+str(j)+'\t'+str(k)+'\t'+str(p)+'\t'+str(r)+'\t'+str(f)+'\t'+str(s))  #replace with write to file
            file.write('\n')

file.close()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

gs = gridspec.GridSpec(1, len(datasets))

df=pd.read_table('process_crf.txt',sep="\t")

fig = plt.figure()

for i in range(len(datasets)):
    cur = df[df['dataset']==datasets[i]]
    ax = plt.subplot(gs[0,i])
    plt.xlabel(dictionary[datasets[i]])
    plt.xticks(np.arange(1, len(configs), step=3))
    p = []
    r = []
    f = []
    for j in configs:
        con = cur[cur['config']==j]
        y = []
        p.append(con['precision'].mean())
        r.append(con['recall'].mean())
        f.append(con['f1'].mean())
        y.append(con['precision'].mean())
        y.append(con['recall'].mean())
        y.append(con['f1'].mean())
        plt.scatter(np.array([j]*3),np.array(y),marker = "d")
    ax.plot(np.array(configs),np.array(p),linestyle = '-',color = 'black',label = 'precision')
    ax.plot(np.array(configs),np.array(r),linestyle = '--',color = 'black',label = 'recall')
    ax.plot(np.array(configs),np.array(f),linestyle = ':',color = 'black', label = 'f1')
    ax.legend()
    fig.add_subplot(ax)
plt.show()

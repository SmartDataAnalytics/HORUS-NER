import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

gs = gridspec.GridSpec(1, 4)

df=pd.read_table('process.txt',sep=" ")

datasets = list(set(df['dataset']))

fig = plt.figure()

for i in range(len(datasets)-1,-1,-1):
    cur = df[df['dataset']==datasets[i]]
    ax = plt.subplot(gs[0,i])
    plt.xticks(np.array(range(1,11)),np.array(range(1,11)))
    p = []
    r = []
    f = []
    for j in range(1,11):
        con = cur[cur['config']==j]
        y = []
        p.append(con['precision'].mean())
        r.append(con['recall'].mean())
        f.append(con['f1'].mean())
        y.append(con['precision'].mean())
        y.append(con['recall'].mean())
        y.append(con['f1'].mean())
        plt.scatter(np.array([j]*3),np.array(y),marker = "d")
    ax.plot(np.array(range(1,11)),np.array(p),linestyle = ':',color = 'black',label = 'precision')
    ax.plot(np.array(range(1,11)),np.array(r),linestyle = '--',color = 'black',label = 'recall')
    ax.plot(np.array(range(1,11)),np.array(f),linestyle = '-.',color = 'black', label = 'f1')
    ax.legend()
    fig.add_subplot(ax)
plt.show()

'''
        p = con[con['precision'] == con['precision'].max()]['label'].to_string(index=False)
        f = con[con['f1'] == con['f1'].max()]['label'].to_string(index=False)
        s = con[con['support'] == con['support'].max()]['label'].to_string(index=False)
        r = con[con['recall'] == con['recall'].max()]['label'].to_string(index=False)
        print(datasets[i],j,'precision = '+p,'recall = '+r,'f1 = '+f,'support = '+s)    #replace with write to file
'''

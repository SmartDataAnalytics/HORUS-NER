import pandas as pd

import sys

df=pd.read_table('plot',sep="\t")

df = df[df['cross-validation'] == True]

datasets = list(set(df['dataset1']))


for i in datasets:
    cur = df[df['dataset1']==i]
    for j in range(1,11):
        conf = cur[cur['config']==j]
        labels = ['PER','ORG','LOC']
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
            print(i,j,k,p,r,f,s)  #replace with write to file

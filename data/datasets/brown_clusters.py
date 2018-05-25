import pickle

from src.config import HorusConfig

config = HorusConfig()

def conll2sentence():
    f_out = open(config.dir_datasets + 'all_sentences.txt', 'w+')
    for file in ['Ritter/ner.txt',
                 'wnut/2015.conll.freebase',
                 'wnut/2016.conll.freebase',
                 'wnut/emerging.test.annotated']:
        filepath= config.dir_datasets + file
        sentence=''
        with open(filepath) as f:
            content = f.readlines()
            for x in content:
                if x!='\n':
                    sentence+=x.split('\t')[0] + ' '
                else:
                    f_out.write(sentence.strip() + '\n')
                    sentence=''
        f_out.flush()

    f_out.close()

def browncluster2dict():
    try:
        brown = dict()
        with open(config.dir_datasets + 'output_brownclusters.txt') as f:
            content = f.readlines()
            for x in content:
                n=x.split('\t')
                brown.update({n[1]:str(n[0])})
        with open(config.dir_datasets + 'browclustersdict.pkl', 'wb') as output:
            pickle.dump(brown, output, pickle.HIGHEST_PROTOCOL)
    except:
        raise


browncluster2dict()
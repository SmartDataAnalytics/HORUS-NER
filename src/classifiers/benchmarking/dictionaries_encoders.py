import pickle

import os
import tarfile
import urllib2

import pandas as pd
import sklearn
from nltk import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder

from src.config import HorusConfig
from src.core.util import definitions
from src.core.util.definitions import encoder_int_lemma_name, encoder_int_stem_name, \
    encoder_onehot_words_name, encoder_int_words_name, encoder_onehot_lemma_name, \
    encoder_onehot_stem_name

config = HorusConfig()

def word2dict(experiment_folder, datasets):
    try:
        config.logger.info('creating encoders: ' + datasets)
        experiment_folder+='/'
        wnl = WordNetLemmatizer()
        stemmer = SnowballStemmer("english")
        words=[]
        lemmas=[]
        stemmers=[]

        all_sentences_name='raw_data_brown_clusters.txt'

        f = open(config.dir_encoders + all_sentences_name, 'w+')
        for ds in datasets.split():
            config.logger.info('creating encoder for: ' + ds)
            _file = config.dir_output + experiment_folder + ds
            df = pd.read_csv(_file, delimiter="\t", skiprows=1, header=None, keep_default_na=False,
                             na_values=['_|_'], usecols=[definitions.INDEX_ID_SENTENCE, definitions.INDEX_TOKEN, definitions.INDEX_IS_COMPOUND])
            prev_sent_id=df.iloc[0].at[definitions.INDEX_ID_SENTENCE]
            sentence=''
            for row in df.itertuples():
                w=str(row[2])
                words.append(w.lower())
                try:
                    lemmas.append(wnl.lemmatize(w.lower()))
                except:
                    continue
                try:
                    stemmers.append(stemmer.stem(w.lower()))
                except:
                    continue

                if prev_sent_id==row[1]:
                    if row[3] != 1:
                        sentence += ' ' + w
                else:
                    f.write(sentence.strip() + '\n')
                    sentence=w
                prev_sent_id=row[1]
            f.flush()
        f.close()

        config.logger.info('total tokens: ' + str(len(words)))
        config.logger.info('total lemmas: ' + str(len(lemmas)))
        config.logger.info('total stemmers: ' + str(len(stemmers)))
        words.append('')
        lemmas.append('')
        stemmers.append('')
        words=set(words)
        lemmas=set(lemmas)
        stemmers=set(stemmers)
        config.logger.info('words vocabulary size: ' + str(len(words)))
        config.logger.info('lemmas vocabulary size: ' + str(len(lemmas)))
        config.logger.info('stemmer vocabulary size: ' + str(len(stemmers)))


        for a,b,c in ((words, encoder_int_words_name, encoder_onehot_words_name),
                      (lemmas, encoder_int_lemma_name, encoder_onehot_lemma_name),
                      (stemmers, encoder_int_stem_name, encoder_onehot_stem_name)):
            le = sklearn.preprocessing.LabelEncoder()
            le.fit(list(a))
            #integer_encoded = le.fit_transform(list(a))
            #onehot = OneHotEncoder(sparse=False)
            #integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            #onehot = onehot.fit_transform(integer_encoded)
            joblib.dump(le, config.dir_encoders + b, compress=3)
            #joblib.dump(onehot, config.dir_encoders + b, compress=3)

    except:
        raise

def conll2sentence():
    config.logger.info('creating sentences file for brown clusters')
    f_out = open(config.dir_datasets + 'all_sentences.txt', 'w+')
    datasets = {'Ritter/ner.txt': 0,
               'wnut/2015.conll.freebase': 0,
               'wnut/2016.conll.freebase':0,
               'wnut/emerging.test.annotated':0
               }
    for file, index in datasets.iteritems():
        filepath= config.dir_datasets + file
        sentence=''
        tot_loc=0
        tot_per=0
        tot_org=0
        tot_others=0
        tot_k=0
        tot_sentences=0
        tot_geral=0
        with open(filepath) as f:
            content = f.readlines()
            k=[]
            for x in content:
                if x!='\n':
                    w=x.split('\t')[0]
                    label=x.split('\t')[1].replace('\n','')
                    k.append(label.replace('B-', '').replace('I-', ''))
                    if label in definitions.NER_TAGS_LOC:
                        tot_loc+=1
                    elif label in definitions.NER_TAGS_ORG:
                        tot_org+=1
                    elif label in definitions.NER_TAGS_PER:
                        tot_per+=1
                    elif label != 'O':
                        tot_k +=1
                    else:
                        tot_others+=1
                    tot_geral+=1
                    sentence+=w + ' '
                else:
                    tot_sentences+=1
                    f_out.write(sentence.strip() + '\n')
                    sentence=''
            tot_sentences += 1
            f_out.write(sentence.strip() + '\n')
            print('%s (%d) & %d & %d & %d & %d & %d & %d & %d \\\\' % (file, len(set(k)), tot_sentences, tot_geral, tot_org, tot_loc, tot_per, tot_k, tot_others))
        f_out.flush()

    f_out.close()

def browncluster2dict(filepath, filename):
    try:
        if not os.path.exists(filepath):
            config.logger.info('downloading the data')
            os.makedirs(filepath)
            request = urllib2.urlopen('https://s3-eu-west-1.amazonaws.com/downloads.gate.ac.uk/resources/derczynski-chester-boegh-brownpaths.tar.bz2')
            tar = tarfile.open('derczynski-chester-boegh-brownpaths.tar.bz2', "r:gz")
            tar.extractall()
            tar.close()
        config.logger.info('creating dictionary: ' + filename)
        brown = dict()
        with open(filepath + filename) as f:
            content = f.readlines()
            for x in content:
                n=x.split('\t')
                brown.update({n[1]:str(n[0])})
        with open(config.dir_datasets + '%s_dict.pkl' % (filename), 'wb') as output:
            pickle.dump(brown, output, pickle.HIGHEST_PROTOCOL)
        config.logger.info('file generated')
    except:
        raise




#browncluster2dict('output_brownclusters.txt')

conll2sentence()
exit(0)
word2dict('EXP_004', '2016.conll.freebase.ascii.txt.horus emerging.test.annotated.horus ner.txt.horus 2015.conll.freebase.horus')
exit(0)
browncluster2dict(config.dir_datasets + 'brown_clusters/', 'gha.500M-c1000-p1.paths')
browncluster2dict(config.dir_datasets + 'brown_clusters/', 'gha.64M-c640-p1.paths')
browncluster2dict(config.dir_datasets + 'brown_clusters/', 'gha.64M-c320-p1.paths')

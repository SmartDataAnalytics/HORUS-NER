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
                w=row[2]
                words.append(w.lower())
                try:
                    lemma=wnl.lemmatize(w.lower().decode('utf-8'))
                    lemmas.append(lemma)
                except:
                    continue
                try:
                    steam=stemmer.stem(w.lower().decode('utf-8'))
                    stemmers.append(steam)
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
        with open(filepath + 'paths/' + filename) as f:
            content = f.readlines()
            for x in content:
                n=x.split('\t')
                brown.update({n[1]:str(n[0])})
        with open(config.dir_datasets + '%s_dict.pkl' % (filename), 'wb') as output:
            pickle.dump(brown, output, pickle.HIGHEST_PROTOCOL)
        config.logger.info('file generated')
    except:
        raise

word2dict('EXP_004', '2016.conll.freebase.ascii.txt.horus emerging.test.annotated.horus ner.txt.horus 2015.conll.freebase.horus')
exit(0)
browncluster2dict(config.dir_datasets + 'brown_clusters/', 'gha.500M-c1000-p1.paths')
browncluster2dict(config.dir_datasets + 'brown_clusters/', 'gha.64M-c640-p1.paths')
browncluster2dict(config.dir_datasets + 'brown_clusters/', 'gha.64M-c320-p1.paths')
# -*- coding: utf-8 -*-
import multiprocessing
import os
import pickle

from nltk import WordNetLemmatizer, SnowballStemmer
from sklearn.externals import joblib
import pandas as pd
from nltk.corpus import stopwords
from nltk import LancasterStemmer, re, WordNetLemmatizer
from src.util import definitions
from src.util.definitions import encoder_le1_name, dict_exp_configurations
from src.config import HorusConfig
from src.util.definitions import EXPERIMENT_FOLDER
import numpy as np

def shape(word):
    word_shape = 0 #'other'
    if re.match('[0-9]+(\.[0-9]*)?|[0-9]*\.[0-9]+$', word):
        word_shape = 1 #'number'
    elif re.match('\W+$', word):
        word_shape = 2 #'punct'
    elif re.match('[A-Z][a-z]+$', word):
        word_shape = 3 #'capitalized'
    elif re.match('[A-Z]+$', word):
        word_shape = 4 # 'uppercase'
    elif re.match('[a-z]+$', word):
        word_shape = 5 #'lowercase'
    elif re.match('[A-Z][a-z]+[A-Z][a-z]+[A-Za-z]*$', word):
        word_shape = 6 #'camelcase'
    elif re.match('[A-Za-z]+$', word):
        word_shape = 7 #'mixedcase'
    elif re.match('__.+__$', word):
        word_shape = 8 # 'wildcard'
    elif re.match('[A-Za-z0-9]+\.$', word):
        word_shape = 9 # 'ending-dot'
    elif re.match('[A-Za-z0-9]+\.[A-Za-z0-9\.]+\.$', word):
        word_shape = 10 # 'abbreviation'
    elif re.match('[A-Za-z0-9]+\-[A-Za-z0-9\-]+.*$', word):
        word_shape = 11 #'contains-hyphen'

    return word_shape

def _append_word_lemma_stem(w, l, s):
    t=[]
    try: t.append(enc_word.transform(str(w)))
    except:
        config.logger.warn('enc_word.transform error')
        t.append(0)

    try: t.append(enc_lemma.transform(l.decode('utf-8')))
    except:
        config.logger.warn('enc_lemma.transform error')
        t.append(0)

    try: t.append(enc_stem.transform(s.decode('utf-8')))
    except:
        config.logger.warn('enc_stem.transform error')
        t.append(0)

    return t

def save_configuration_dump_file((_file_path, _file_name, ds, f_key, f_indexes)):
    try:
        config.logger.info(_file_name + ' dump creation starts!')
        X_sentence = [exclude_columns(s, f_indexes) for s in ds[1][0]]
        Y_sentence = [sent2label(s) for s in ds[1][1]]
        X_token = exclude_columns(ds[2][0], f_indexes)
        X_token.replace('O', 0, inplace=True)
        Y_token = [definitions.PLOMNone_label2index[y] for y in ds[2][1]]
        X_crf = [sent2features(s) for s in X_sentence]
        _Y_sentence = np.array([x for s in Y_sentence for x in s])  # trick for scikit_learn on CRF (for the precision_recall_fscore_support method)

        ##X_lstm, y_lstm, max_features, out_size, maxlen = convert_lstm_shape(X_sentence, Y_sentence, f_indexes)
        ##X2_lstm, y2_lstm, max_features_2, out_size_2, maxlen_2 = convert_lstm_shape(ds2[1][0], ds2[1][1], f_indexes)
        ##X1_lstm = pad_sequences(X1_lstm, maxlen=max(maxlen_1, maxlen_2))
        ##y1_lstm = pad_sequences(y1_lstm, maxlen=max(maxlen_1, maxlen_2))
        with open(_file_path, 'wb') as output:
            pickle.dump([_file_name, f_key, X_sentence, Y_sentence, X_token, Y_token,
                         X_crf, _Y_sentence], output, pickle.HIGHEST_PROTOCOL)
        config.logger.info(_file_name + ' created!')
    except Exception as e:
        config.logger.error(repr(e))
        raise e

    return _file_name

def shape_data((horus_m3_path, horus_m4_path)):
    '''
    shape the dataframe, adding further traditional features
    :param file: the horus features file
    :param path: the path
    :param le: the encoder
    :return: an updated dataset and a sentence-shaped dataset
    '''
    try:
        import unicodedata
        from unicodedata import normalize, name

        ds_sentences, y_sentences_shape = [], []
        _sent_temp_feat, _sent_temp_y = [], []

        #fullpath=path+file
        config.logger.info('reading horus features file: ' + horus_m3_path)
        df = pd.read_csv(horus_m3_path, delimiter="\t", skiprows=1, header=None, keep_default_na=False, na_values=['_|_'])
        df = df.drop(df[df[definitions.INDEX_IS_COMPOUND]==1].index)
        oldsentid = df.iloc[0].at[definitions.INDEX_ID_SENTENCE]
        df = df.reset_index(drop=True)
        COLS = len(df.columns)

        import operator
        MAX_ITEM = max(definitions.schemaindex2label.iteritems(), key=operator.itemgetter(0))[0]
        #df = pd.concat([df, pd.DataFrame(columns=range(COLS, (COLS + definitions.STANDARD_FEAT_LEN)))], axis=1)
        df = pd.concat([df, pd.DataFrame(columns=range(COLS, MAX_ITEM + 1))], axis=1)

        config.logger.info(len(df))

        # for test purpose
        stop_next = False

        for row in df.itertuples():
            index=row.Index
            if index % 500 == 0: config.logger.info(index)
            #if index == 15000:
            #    stop_next = True
            if df.loc[index, definitions.INDEX_ID_SENTENCE] != oldsentid:
                ds_sentences.append(_sent_temp_feat)
                _sent_temp_feat = []
                y_sentences_shape.append(_sent_temp_y)
                _sent_temp_y = []
                if stop_next:
                    break

            idsent = df.loc[index].at[definitions.INDEX_ID_SENTENCE]
            token = df.loc[index].at[definitions.INDEX_TOKEN]

            if not isinstance(token, unicode):
                token = unicodedata.normalize('NFKD', unicode(token, 'utf-8')).encode('ascii','ignore')
                if len(token) == 0:
                    token = "'"
            #    a = 1
            #if token == u'´':
            #    token = "'"
            token = token.decode('utf8', 'ignore')


            brown_1000_path = '{:<016}'.format(dict_brown_c1000.get(token, '0000000000000000'))
            brown_640_path = '{:<016}'.format(dict_brown_c640.get(token, '0000000000000000'))
            brown_320_path = '{:<016}'.format(dict_brown_c320.get(token, '0000000000000000'))

            brown_1000=[]
            k=1
            tot_slide=5 #range(len(brown_1000_path)-1)
            for i in range(tot_slide):
                brown_1000.append(brown_1000_path[:k])
                k+=1
            brown_640 = []
            k = 1
            for i in range(tot_slide):
                brown_640.append(brown_640_path[:k])
                k += 1
            brown_320 = []
            k = 1
            for i in range(tot_slide):
                brown_320.append(brown_320_path[:k])
                k += 1

            #if index > 1: prev_prev_serie = df.loc[index-2]
            #if index > 0: prev_serie = df.loc[index-1]
            #if index + 1 < len(df): next_serie = df.loc[index+1]
            #if index + 2 < len(df): next_next_serie = df.loc[index+2]

            _t=[]
            # standard features
            '''
            if index > 0 and prev_serie.at[definitions.INDEX_ID_SENTENCE] == idsent:
                prev_pos = prev_serie.at[definitions.INDEX_POS]
                prev_pos_uni = prev_serie.at[definitions.INDEX_POS_UNI]
                prev_token = prev_serie.at[definitions.INDEX_TOKEN]
                prev_one_char_token = int(len(prev_token) == 1)
                prev_special_char = int(len(re.findall('(http://\S+|\S*[^\w\s]\S*)', prev_token)) > 0)
                prev_first_capitalized = int(prev_token[0].isupper())
                prev_capitalized = int(prev_token.isupper())
                prev_title = int(prev_token.istitle())
                prev_digit = int(prev_token.isdigit())
                prev_stop_words = int(prev_token in stop)
                prev_small = int(len(prev_token) <= 2)
                prev_hyphen = int('-' in prev_token)
                prev_sh = shape(prev_token)
                try: prev_lemma = wnl.lemmatize(prev_token.lower())
                except: prev_lemma = ''.decode('utf8')
                try: prev_stem = stemmer.stem(prev_token.lower())
                except: prev_stem = ''.decode('utf8')
            else:
                prev_pos = ''
                prev_pos_uni = ''
                prev_token = ''
                prev_lemma = ''
                prev_stem = ''
                prev_one_char_token = -1
                prev_special_char = -1
                prev_first_capitalized = -1
                prev_capitalized = -1
                prev_title = -1
                prev_digit = -1
                prev_stop_words = -1
                prev_small = -1
                prev_hyphen = -1
                prev_sh = -1

            if index > 1 and prev_prev_serie.at[definitions.INDEX_ID_SENTENCE] == idsent:
                prev_prev_pos = prev_prev_serie.at[definitions.INDEX_POS]
                prev_prev_pos_uni = prev_prev_serie.at[definitions.INDEX_POS_UNI]
                prev_prev_token = prev_prev_serie.at[definitions.INDEX_TOKEN]
                prev_prev_one_char_token = int(len(prev_prev_token) == 1)
                prev_prev_special_char = int(len(re.findall('(http://\S+|\S*[^\w\s]\S*)', prev_prev_token)) > 0)
                prev_prev_first_capitalized = int(prev_prev_token[0].isupper())
                prev_prev_capitalized = int(prev_prev_token.isupper())
                prev_prev_title = int(prev_prev_token.istitle())
                prev_prev_digit = int(prev_prev_token.isdigit())
                prev_prev_stop_words = int(prev_prev_token in stop)
                prev_prev_small = int(len(prev_prev_token) <= 2)
                prev_prev_hyphen = int('-' in prev_prev_token)
                prev_prev_sh = shape(prev_prev_token)
                try: prev_prev_lemma = wnl.lemmatize(prev_prev_token.lower())
                except: prev_prev_lemma = ''
                try: prev_prev_stem = stemmer.stem(prev_prev_token.lower())
                except: prev_prev_stem = ''
            else:
                prev_prev_pos= ''
                prev_prev_pos_uni = ''
                prev_prev_token= ''
                prev_prev_lemma=''
                prev_prev_stem=''
                prev_prev_one_char_token= -1
                prev_prev_special_char= -1
                prev_prev_first_capitalized= -1
                prev_prev_capitalized= -1
                prev_prev_title= -1
                prev_prev_digit= -1
                prev_prev_stop_words= -1
                prev_prev_small= -1
                prev_prev_hyphen= -1
                prev_prev_sh= -1

            if index + 1 < len(df) and next_serie.at[definitions.INDEX_ID_SENTENCE] == idsent:
                next_pos = next_serie.at[definitions.INDEX_POS]
                next_pos_uni = next_serie.at[definitions.INDEX_POS_UNI]
                next_token = next_serie.at[definitions.INDEX_TOKEN]
                next_one_char_token = int(len(next_token) == 1)
                next_special_char = int(len(re.findall('(http://\S+|\S*[^\w\s]\S*)', next_token)) > 0)
                next_first_capitalized = int(next_token[0].isupper())
                next_capitalized = int(next_token.isupper())
                next_title = int(next_token.istitle())
                next_digit = int(next_token.isdigit())
                next_stop_words = int(next_token in stop)
                next_small = int(len(next_token) <= 2)
                next_hyphen = int('-' in next_token)
                next_sh = shape(next_token)
                try: next_lemma = wnl.lemmatize(next_token.lower())
                except: next_lemma = ''
                try: next_stem = stemmer.stem(next_token.lower())
                except: next_stem = ''
            else:
                next_pos = ''
                next_pos_uni = ''
                next_token = ''
                next_lemma=''
                next_stem=''
                next_one_char_token = -1
                next_special_char = -1
                next_first_capitalized = -1
                next_capitalized = -1
                next_title = -1
                next_digit = -1
                next_stop_words = -1
                next_small = -1
                next_hyphen = -1
                next_sh = -1

            if index + 2 < len(df) and next_next_serie.at[definitions.INDEX_ID_SENTENCE] == idsent:
                next_next_pos = next_next_serie.at[definitions.INDEX_POS]
                next_next_pos_uni = next_next_serie.at[definitions.INDEX_POS_UNI]
                next_next_token = next_next_serie.at[definitions.INDEX_TOKEN]
                next_next_one_char_token = int(len(next_next_token) == 1)
                next_next_special_char = int(len(re.findall('(http://\S+|\S*[^\w\s]\S*)', next_next_token)) > 0)
                next_next_first_capitalized = int(next_next_token[0].isupper())
                next_next_capitalized = int(next_next_token.isupper())
                next_next_title = int(next_next_token.istitle())
                next_next_digit = int(next_next_token.isdigit())
                next_next_stop_words = int(next_next_token in stop)
                next_next_small = int(len(next_next_token) <= 2)
                next_next_hyphen = int('-' in next_next_token)
                next_next_sh = shape(next_next_token)
                try: next_next_lemma = wnl.lemmatize(next_next_token.lower())
                except: next_next_lemma = ''
                try: next_next_stem = stemmer.stem(next_next_token.lower())
                except: next_next_stem = ''
            else:
                next_next_pos = ''
                next_next_pos_uni = ''
                next_next_token = ''
                next_next_lemma=''
                next_next_stem=''
                next_next_one_char_token = -1
                next_next_special_char = -1
                next_next_first_capitalized = -1
                next_next_capitalized = -1
                next_next_title = -1
                next_next_digit = -1
                next_next_stop_words = -1
                next_next_small = -1
                next_next_hyphen = -1
                next_next_sh = -1
            
            
            # standard features [t-2, t-1, t, t+1, t+2] (12*5=60)
            _t.extend([le.transform(prev_prev_pos),
                       le.transform(prev_prev_pos_uni),
                       prev_prev_one_char_token,
                       prev_prev_special_char,
                       prev_prev_first_capitalized,
                       prev_prev_capitalized,
                       prev_prev_title,
                       prev_prev_digit,
                       prev_prev_stop_words,
                       prev_prev_small,
                       prev_prev_hyphen,
                       prev_prev_sh])
            _t.extend([le.transform(prev_pos),
                       le.transform(prev_pos_uni),
                       prev_one_char_token,
                       prev_special_char,
                       prev_first_capitalized,
                       prev_capitalized,
                       prev_title,
                       prev_digit,
                       prev_stop_words,
                       prev_small,
                       prev_hyphen,
                       prev_sh])
            '''

            # word, lemma, stem for [t-2, t-1, t, t+1, t+2] (3*5=15)
            lemma = ''
            stem = ''
            try: lemma = lemmatize(token.lower())
            except: pass
            try: stem = stemo(token.lower())
            except: pass

            _t.extend([token.lower(), lemma.decode('utf-8'), stem.decode('utf-8')])

            # _t.extend(_append_word_lemma_stem(prev_prev_token.lower(), prev_prev_lemma, prev_prev_stem))
            # _t.extend(_append_word_lemma_stem(prev_token.lower(), prev_lemma, prev_stem))
            _t.extend(_append_word_lemma_stem(token.lower(), lemma, stem))
            # _t.extend(_append_word_lemma_stem(next_token.lower(), next_lemma, next_stem))
            # _t.extend(_append_word_lemma_stem(next_next_token.lower(), next_next_lemma, next_next_stem))

            # 12
            _t.extend([le.transform(df.loc[index].at[definitions.INDEX_POS]),
                       le.transform(df.loc[index].at[definitions.INDEX_POS_UNI]),
                       (len(token) == 1),
                       (len(re.findall('(http://\S+|\S*[^\w\s]\S*)', token)) > 0),
                       (token[0].isupper()),
                       (token.isupper()),
                       (token.istitle()),
                       (token.isdigit()),
                       (token in stop),
                       (len(token) <= 2),
                       ('-' in token),
                       shape(token)])
            '''
            _t.extend([le.transform(next_pos),
                       le.transform(next_pos_uni),
                       next_one_char_token,
                       next_special_char,
                       next_first_capitalized,
                       next_capitalized,
                       next_title,
                       next_digit,
                       next_stop_words,
                       next_small,
                       next_hyphen,
                       next_sh])
            _t.extend([le.transform(next_next_pos),
                       le.transform(next_next_pos_uni),
                       next_next_one_char_token,
                       next_next_special_char,
                       next_next_first_capitalized,
                       next_next_capitalized,
                       next_next_title,
                       next_next_digit,
                       next_next_stop_words,
                       next_next_small,
                       next_next_hyphen,
                       next_next_sh])
            '''

            # brown clusters [320, 640, 1000] (5*3=15)
            _t.extend(brown_320)
            _t.extend(brown_640)
            _t.extend(brown_1000)

            df.iloc[[index], COLS:(COLS + MAX_ITEM + 1)] = _t

            # NER class
            y = int(df.loc[index].at[definitions.INDEX_TARGET_NER])


            oldsentid = idsent

            _sent_temp_feat.append(df.loc[index])
            _sent_temp_y.append(y)

        y_tokens_shape = df[definitions.INDEX_TARGET_NER].copy()
        del df[definitions.INDEX_TARGET_NER]

        config.logger.info('total of sentences: ' + str(len(ds_sentences)))
        config.logger.info('total of tokens: ' + str(len(df)))

        data = file, (ds_sentences, y_sentences_shape), (df, y_tokens_shape)
        with open(horus_m4_path, 'wb') as output:
            pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)
        config.logger.info('file exported: ' + horus_m4_path)


    except Exception as e:
        config.logger.error(repr(e))


def extract_lexical_and_shape_data():
    job_args = []
    for ds in definitions.NER_DATASETS:
        horus_m3_path = ds[1] + ds[2].replace('.horusx', '.horus3')
        config.logger.info(horus_m3_path)
        if not os.path.isfile(horus_m3_path):
            config.logger.error(' -- file .horus3 does not exist!')
            config.logger.error(' -- please check the file extract_cv_tx.py to create it!')
        else:
            horus_m4_path = horus_m3_path.replace('.horus3', '.horus4')
            #_file = config.dir_output + experiment_folder + '_' + ds + '_shaped.pkl'
            if os.path.isfile(horus_m4_path):
                config.logger.warn('file already exists: %s' % (horus_m4_path))
            else:
                job_args.append((horus_m3_path, horus_m4_path))

    config.logger.info('job args created: ' + str(len(job_args)))
    if len(job_args) > 0:
        p = multiprocessing.Pool(multiprocessing.cpu_count())
        asyncres = p.map(shape_data, job_args)
        #config.logger.info(len(asyncres))
        #onfig.logger.info('done: ' + str(len(asyncres)) + ' files exported!')

if __name__ == "__main__":

    config = HorusConfig()

    config.logger.info('loading lemmatizers')
    stemmer = SnowballStemmer('english')
    stop = set(stopwords.words('english'))
    wnl = WordNetLemmatizer()

    from functools32 import lru_cache
    lemmatize = lru_cache(maxsize=50000)(wnl.lemmatize)
    stemo = lru_cache(maxsize=50000)(stemmer.stem)

    config.logger.info('loading encoders')
    enc_le1 = joblib.load(config.dir_encoders + definitions.encoder_le1_name)
    enc_le2 = joblib.load(config.dir_encoders + definitions.encoder_le2_name)
    enc_word = joblib.load(config.dir_encoders + definitions.encoder_int_words_name)
    enc_lemma = joblib.load(config.dir_encoders + definitions.encoder_int_lemma_name)
    enc_stem = joblib.load(config.dir_encoders + definitions.encoder_int_stem_name)
    le = joblib.load(config.dir_encoders + encoder_le1_name)

    config.logger.info('loading brown corpus')
    dict_brown_c1000 = joblib.load(config.dir_datasets + 'gha.500M-c1000-p1.paths_dict.pkl')
    dict_brown_c640 = joblib.load(config.dir_datasets + 'gha.64M-c640-p1.paths_dict.pkl')
    dict_brown_c320 = joblib.load(config.dir_datasets + 'gha.64M-c320-p1.paths_dict.pkl')
    #_SET_MASK = '_%s_config_%s.pkl'
    #datasets = '2015.conll.freebase.horus 2016.conll.freebase.ascii.txt.horus ner.txt.horus emerging.test.annotated.horus'

    try:
        config.logger.info('extracting final feature files (lexical + cv + tx)')
        # ds_name, (X1, y1 [DT-shape]), (X2, y2 [CRF-shape]), (X3, y3 [NN-shape])
        extract_lexical_and_shape_data()

        '''
        # 
        # -- REMOVING THIS UNECESSARY STEP --
        # -- THIS CREATES LOT OF REDUNDANT DATA --
        #
        config.logger.info('creating the dump files: configurations x datasets')
        # multithread to shape the datasets for all configurations in parallel
        job_args = []
        set_file_dump_names = []
        for ds in shaped_datasets:
            for f_key, f_indexes in dict_exp_configurations.iteritems():
                _set_name = _SET_MASK % (ds[0], str(f_key))
                _file = config.dir_output + EXPERIMENT_FOLDER + _set_name
                if os.path.isfile(_file) is False:
                    job_args.append((_file, _set_name, ds, f_key, f_indexes))
                set_file_dump_names.append(_set_name)

        config.logger.info('job args created - config dumps: ' + str(len(job_args)))
        if len(job_args) > 0:
            config.logger.info('creating dump files...')
            p = multiprocessing.Pool(multiprocessing.cpu_count())
            p.map(save_configuration_dump_file, job_args)
        
        '''

    except Exception as e:
        config.logger.error(repr(e))

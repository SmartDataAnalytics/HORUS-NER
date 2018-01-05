import sys
import pandas as pd
import numpy as np
import logging
from horus.core import definitions
from nltk.corpus import stopwords
from nltk import LancasterStemmer, re

lancaster_stemmer = LancasterStemmer()
stop = set(stopwords.words('english'))

loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def horus_to_features(horusfile, le):
    print horusfile
    features, sentence_shape = [], []
    targets, tokens_shape, y_sentences_shape, y_tokens_shape = [], [], [], []

    df = pd.read_csv(horusfile, delimiter=",", skiprows=1, header=None, keep_default_na=False, na_values=['_|_'])
    oldsentid = df.get_values()[0][1]
    for index, linha in df.iterrows():
        if len(linha)>0:
            if linha[7] == 0: #no compounds
                if linha[1] != oldsentid:
                    sentence_shape.append(features)
                    y_sentences_shape.append(targets)
                    targets, features = [], []

                idsent = linha[1]
                idtoken = linha[2]
                pos_bef = ''
                pos_aft = ''
                if index > 0 and df.get_value(index-1, 7) == 0:
                    pos_bef = df.get_value(index-1,5)
                if index + 1 < len(df) and df.get_value(index+1, 7) == 0:
                    pos_aft = df.get_value(index+1,5)
                token = linha[3]
                postag = linha[5]
                one_char_token = len(token) == 1
                special_char = len(re.findall('(http://\S+|\S*[^\w\s]\S*)', token)) > 0
                first_capitalized = token[0].isupper()
                capitalized = token.isupper()
                title = token.istitle()
                digit = token.isdigit()
                stop_words = token in stop
                small = True if len(horusfile[3]) <= 2 else False
                stemmer_lanc = lancaster_stemmer.stem(token)
                nr_images_returned = linha[17]
                nr_websites_returned = linha[25]
                hyphen = '-' in token
                cv_loc = float(linha[12])
                cv_org = float(linha[13])
                cv_per = float(linha[14])
                cv_dist = float(linha[15])
                cv_plc = float(linha[16])
                tx_loc = float(linha[20])
                tx_org = float(linha[21])
                tx_per = float(linha[22])
                tx_err = float(linha[23])
                tx_dist = float(linha[24])

                if linha[6] in definitions.NER_TAGS_LOC: ner = u'LOC'
                elif linha[6] in definitions.NER_TAGS_ORG: ner = u'ORG'
                elif linha[6] in definitions.NER_TAGS_PER: ner = u'PER'
                else: ner = u'O'

                #standard shape
                sel_features = [idsent, idtoken, token, token.lower(), stemmer_lanc,
                                pos_bef, postag, pos_aft, definitions.KLASSES2[ner],
                                le.transform(pos_bef), le.transform(postag), le.transform(pos_aft),
                                title, digit, one_char_token, special_char, first_capitalized,
                                hyphen, capitalized, stop_words, small,
                                nr_images_returned, nr_websites_returned,
                                cv_org, cv_loc, cv_per, cv_dist, cv_plc,
                                tx_org, tx_loc, tx_per, tx_dist, tx_err]

                features.append(sel_features)

                if linha[51] in definitions.NER_TAGS_LOC: y = u'LOC'
                elif linha[51] in definitions.NER_TAGS_ORG: y = u'ORG'
                elif linha[51] in definitions.NER_TAGS_PER: y = u'PER'
                else: y = u'O'

                targets.append(y)

                tokens_shape.append(sel_features[9:len(sel_features)])
                y_tokens_shape.append(definitions.KLASSES2[y])

                oldsentid = linha[1]

    print 'total of sentences', len(sentence_shape)
    print 'total of tokens', len(tokens_shape)
    return sentence_shape, y_sentences_shape, tokens_shape, y_tokens_shape

def get_remaining_column_indexes(line, cols):
    print("excluding irrelevant indexes...")
    to_exclude = []
    for i in range(len(line)):
        if (i not in cols) and (i != definitions.HORUS_FORMAT_INDEX_COL_TARGET_NER) \
                and (i != definitions.HORUS_FORMAT_INDEX_COL_ID_SENTENCE) \
                and (i != definitions.HORUS_FORMAT_INDEX_COL_ID_WORD):
            to_exclude.append(i)
    print("total of columns to exclude: ", str(len(to_exclude)))
    print("total of columns to keep: ", str(len(cols) + 1))
    return to_exclude

def convertHORUStoCoNLL(dataset, cols, out_path):
    '''
    read a horus format file and converts it to CoNLL standard format
    :param dataset: the horus format data set
    :param cols: the column indexes to be converted
    :return:
    '''
    print("starting data format conversion")
    y = []
    shouldleave = False
    dt_conll_format = pd.DataFrame()
    i_sentence = 1
    df = pd.read_csv(dataset, delimiter=",", skiprows=1, header=None, keep_default_na=False, na_values=['_|_'])
    print("dataset shape: ", df.shape)
    # get the columns that should be excluded
    columns_to_exclude = get_remaining_column_indexes(df.iloc[0], cols)
    # removing compounds data
    df.drop(df[df[definitions.HORUS_FORMAT_INDEX_COL_IS_COMPOUND] == 1].index, inplace=True)
    print("dataset shape: ", df.shape)
    # removing empty lines
    df.dropna(how="all", inplace=True)
    print("dataset shape: ", df.shape)
    # get targets before excluding
    y = df.iloc[:,definitions.HORUS_FORMAT_INDEX_COL_TARGET_NER].copy()
    oldsentid = df.get_values()[0][definitions.HORUS_FORMAT_INDEX_COL_ID_SENTENCE]
    # exclude irrelevant columns
    df.drop(df.columns[columns_to_exclude], axis=1, inplace=True)
    print("dataset shape: ", df.shape)
    for index, linha in df.iterrows():
        if len(linha) > 0:
            if linha[definitions.HORUS_FORMAT_INDEX_COL_ID_SENTENCE] == oldsentid:
                dt_conll_format = dt_conll_format.append(linha)
            else:
                linhanan = linha.copy()
                linhanan.loc[:] = np.nan
                if index + 1 < len(df.index):
                    oldsentid = df.get_value(index, definitions.HORUS_FORMAT_INDEX_COL_ID_SENTENCE)
                    i_sentence = i_sentence + 1
                dt_conll_format = dt_conll_format.append(linhanan)
                dt_conll_format = dt_conll_format.append(linha)

    # excluding auxiliar id sentence column
    dt_conll_format.drop(dt_conll_format.columns[0], axis=1, inplace=True)

    print("dataset shape: ", df.shape)
    print("done! %s sentences exported to ConLL format!", str(i_sentence))
    dt_conll_format.to_csv(path_or_buf=out_path, sep="\t", index=False, header=False, float_format='%.0f')
    print("file saved to: ", out_path)
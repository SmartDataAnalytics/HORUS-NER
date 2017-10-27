import sys
import pandas as pd
import numpy as np
import logging
from horus.core import definitions
from horus.core.config import HorusConfig

config = HorusConfig()

# :: Logging level ::
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


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

features = [3,4,5,6,11,12,13,14,15,16,17,19,20,21,22,24,25]
convertHORUStoCoNLL(config.output_path + "/experiments/EXP_do_tokenization/out_exp003_coNLL2003testA_en_NLTK.csv", features,
                    config.output_path + "/experiments/EXP_do_tokenization/conversion/out_exp003_coNLL2003testA_en_NLTK.csv")

convertHORUStoCoNLL(config.output_path + "/experiments/EXP_do_tokenization/out_exp003_ritter_en_tweetNLP.csv", features,
                    config.output_path + "/experiments/EXP_do_tokenization/conversion/out_exp003_ritter_en_tweetNLP.csv")

convertHORUStoCoNLL(config.output_path + "/experiments/EXP_do_tokenization/out_exp003_wnut15_en_tweetNLP.csv", features,
                    config.output_path + "/experiments/EXP_do_tokenization/conversion/out_exp003_wnut15_en_tweetNLP.csv")

convertHORUStoCoNLL(config.output_path + "/experiments/EXP_do_tokenization/out_exp003_wnut16_en_tweetNLP.csv", features,
                    config.output_path + "/experiments/EXP_do_tokenization/conversion/out_exp003_wnut16_en_tweetNLP.csv")
#if __name__ == "__main__":
#    if len(sys.argv) != 2:
#        print "please inform: 1: data set and 2: column indexes ([1, .., n])"
#    else:
#        convertHORUStoCoNLL(sys.argv[0], sys.argv[1])

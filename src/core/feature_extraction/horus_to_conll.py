import sys
import pandas as pd
import numpy as np
import logging
from src.core.util import definitions
from nltk.corpus import stopwords
from nltk import LancasterStemmer, re
import argparse
from src.config import HorusConfig

lancaster_stemmer = LancasterStemmer()
stop = set(stopwords.words('english'))
config = HorusConfig()

def get_features(horusfile, le):
    '''
    converts horus features file to algorithm's expected shapes,
    adding further traditional features
    :param horusfile: the horus features file
    :param le: the encoder
    :return: a (standard matrix + a CRF + a LSTM) file formats
    '''
    features, sentence_shape = [], []
    targets, tokens_shape, y_sentences_shape, y_tokens_shape = [], [], [], []

    config.logger.info('reading horus features file: ' + horusfile)
    df = pd.read_csv(horusfile, delimiter="\t", skiprows=1, header=None, keep_default_na=False, na_values=['_|_'])
    oldsentid = df.get_values()[0][definitions.INDEX_ID_SENTENCE]
    for index, feat in df.iterrows():
        if len(feat)>0:
            if feat[definitions.INDEX_IS_COMPOUND] == 0: #no compounds
                if feat[definitions.INDEX_ID_SENTENCE] != oldsentid:
                    sentence_shape.append(features)
                    y_sentences_shape.append(targets)
                    targets, features = [], []

                idsent = feat[definitions.INDEX_ID_SENTENCE]
                idtoken = feat[definitions.INDEX_ID_WORD]
                pos_bef = ''
                pos_aft = ''
                if index > 0 and df.get_value(index-1, 7) == 0:
                    pos_bef = df.get_value(index-1,5)
                if index + 1 < len(df) and df.get_value(index+1, 7) == 0:
                    pos_aft = df.get_value(index+1,5)
                token = feat[definitions.INDEX_TOKEN]
                postag = feat[definitions.INDEX_POS]
                one_char_token = len(token) == 1
                special_char = len(re.findall('(http://\S+|\S*[^\w\s]\S*)', token)) > 0
                first_capitalized = token[0].isupper()
                capitalized = token.isupper()
                title = token.istitle()
                digit = token.isdigit()
                stop_words = token in stop
                small = True if len(horusfile[3]) <= 2 else False
                stemmer_lanc = lancaster_stemmer.stem(token)
                nr_images_returned = feat[definitions.INDEX_NR_RESULTS_SE_IMG]
                nr_websites_returned = feat[definitions.INDEX_NR_RESULTS_SE_TX]
                hyphen = '-' in token
                cv_loc = float(feat[definitions.INDEX_TOT_CV_LOC])
                cv_org = float(feat[definitions.INDEX_TOT_CV_ORG])
                cv_per = float(feat[definitions.INDEX_TOT_CV_PER])
                cv_dist = float(feat[definitions.INDEX_DIST_CV_I])
                cv_plc = float(feat[definitions.INDEX_PL_CV_I])
                tx_loc = float(feat[definitions.INDEX_TOT_TX_LOC])
                tx_org = float(feat[definitions.INDEX_TOT_TX_ORG])
                tx_per = float(feat[definitions.INDEX_TOT_TX_PER])
                tx_err = float(feat[definitions.INDEX_TOT_ERR_TRANS])
                tx_dist = float(feat[definitions.INDEX_DIST_TX_I])

                if feat[definitions.INDEX_NER] in definitions.NER_TAGS_LOC: ner = u'LOC'
                elif feat[definitions.INDEX_NER] in definitions.NER_TAGS_ORG: ner = u'ORG'
                elif feat[definitions.INDEX_NER] in definitions.NER_TAGS_PER: ner = u'PER'
                else: ner = u'O'

                #standard shape
                f = [idsent, idtoken, token, token.lower(), stemmer_lanc,
                                pos_bef, postag, pos_aft, definitions.KLASSES2[ner],
                                le.transform(pos_bef), le.transform(postag), le.transform(pos_aft),
                                title, digit, one_char_token, special_char, first_capitalized,
                                hyphen, capitalized, stop_words, small,
                                nr_images_returned, nr_websites_returned,
                                cv_org, cv_loc, cv_per, cv_dist, cv_plc,
                                tx_org, tx_loc, tx_per, tx_dist, tx_err,
                                float(feat[definitions.INDEX_TOT_TX_LOC_TM_CNN]),
                                float(feat[definitions.INDEX_TOT_TX_ORG_TM_CNN]),
                                float(feat[definitions.INDEX_TOT_TX_PER_TM_CNN]),
                                float(feat[definitions.INDEX_DIST_TX_I_TM_CNN]),
                                float(feat[definitions.INDEX_TOT_CV_LOC_1_CNN]),
                                float(feat[definitions.INDEX_TOT_CV_ORG_CNN]),
                                float(feat[definitions.INDEX_TOT_CV_PER_CNN]),
                                float(feat[definitions.INDEX_TOT_CV_LOC_2_CNN]),
                                0 if feat[definitions.INDEX_TOT_CV_LOC_3_CNN] == 'O' else float(feat[definitions.INDEX_TOT_CV_LOC_3_CNN]),
                                0 if feat[definitions.INDEX_TOT_CV_LOC_4_CNN] == 'O' else float(feat[definitions.INDEX_TOT_CV_LOC_4_CNN]),
                                0 if feat[definitions.INDEX_TOT_EMB_SIMILAR_LOC] == 'O' else float(feat[definitions.INDEX_TOT_EMB_SIMILAR_LOC]),
                                0 if feat[definitions.INDEX_TOT_EMB_SIMILAR_ORG] == 'O' else float(feat[definitions.INDEX_TOT_EMB_SIMILAR_ORG]),
                                0 if feat[definitions.INDEX_TOT_EMB_SIMILAR_PER] == 'O' else float(feat[definitions.INDEX_TOT_EMB_SIMILAR_PER]),
                                0 if feat[definitions.INDEX_TOT_CV_LOC_5_CNN] == 'O' else float(feat[definitions.INDEX_TOT_CV_LOC_5_CNN]),
                                0 if feat[definitions.INDEX_TOT_CV_LOC_6_CNN] == 'O' else float(feat[definitions.INDEX_TOT_CV_LOC_6_CNN]),
                                0 if feat[definitions.INDEX_TOT_CV_LOC_7_CNN] == 'O' else float(feat[definitions.INDEX_TOT_CV_LOC_7_CNN]),
                                0 if feat[definitions.INDEX_TOT_CV_LOC_8_CNN] == 'O' else float(feat[definitions.INDEX_TOT_CV_LOC_8_CNN]),
                                0 if feat[definitions.INDEX_TOT_EMB_SIMILAR_NONE] == 'O' else float(feat[definitions.INDEX_TOT_EMB_SIMILAR_NONE]),
                                0 if feat[definitions.INDEX_TOT_TX_NONE_TM_CNN] == 'O' else float(feat[definitions.INDEX_TOT_TX_NONE_TM_CNN])
                                ]

                features.append(f)

                if feat[definitions.INDEX_TARGET_NER] in definitions.NER_TAGS_LOC: y = u'LOC'
                elif feat[definitions.INDEX_TARGET_NER] in definitions.NER_TAGS_ORG: y = u'ORG'
                elif feat[definitions.INDEX_TARGET_NER] in definitions.NER_TAGS_PER: y = u'PER'
                else: y = u'O'

                targets.append(y)

                tokens_shape.append(f[9:len(f)])
                y_tokens_shape.append(definitions.KLASSES2[y])

                oldsentid = feat[1]

    config.logger.info('total of sentences: ' + str(len(sentence_shape)))
    config.logger.info('total of tokens: ' + str(len(tokens_shape)))
    return sentence_shape, y_sentences_shape, tokens_shape, y_tokens_shape

def __get_remaining_column_indexes(line, cols):
    print("excluding irrelevant indexes...")
    to_exclude = []
    for i in range(len(line)):
        if (i not in cols) and (i != definitions.INDEX_TARGET_NER) \
                and (i != definitions.INDEX_ID_SENTENCE) \
                and (i != definitions.INDEX_ID_WORD):
            to_exclude.append(i)
    print("total of columns to exclude: ", str(len(to_exclude)))
    print("total of columns to keep: ", str(len(cols) + 1))
    return to_exclude

def convert(experiment_folder, input_dataset_file, features_ind=definitions.FEATURES_INDEX):
    '''
    read a horus format file and converts it to CoNLL standard format
    :param dataset: the horus format data set
    :param cols: the column indexes to be converted
    :return:
    '''
    config.logger.info("starting data format conversion")
    dataset=config.dir_output + experiment_folder + "/" + input_dataset_file

    y = []
    shouldleave = False
    dt_conll_format = pd.DataFrame()
    i_sentence = 1
    df = pd.read_csv(dataset, delimiter="\t", skiprows=1, header=None, keep_default_na=False, na_values=['_|_'])
    print("dataset shape: ", df.shape)
    # get the columns that should be excluded
    columns_to_exclude = __get_remaining_column_indexes(df.iloc[0], features_ind)
    # removing compounds data
    df.drop(df[df[definitions.INDEX_IS_COMPOUND] == 1].index, inplace=True)
    print("dataset shape: ", df.shape)
    # removing empty lines
    df.dropna(how="all", inplace=True)
    print("dataset shape: ", df.shape)
    # get targets before excluding
    y = df.iloc[:, definitions.INDEX_TARGET_NER].copy()
    oldsentid = df.get_values()[0][definitions.INDEX_ID_SENTENCE]
    # exclude irrelevant columns
    df.drop(df.columns[columns_to_exclude], axis=1, inplace=True)
    print("dataset shape: ", df.shape)
    for index, linha in df.iterrows():
        if len(linha) > 0:
            if linha[definitions.INDEX_ID_SENTENCE] == oldsentid:
                dt_conll_format = dt_conll_format.append(linha)
            else:
                linhanan = linha.copy()
                linhanan.loc[:] = np.nan
                if index + 1 < len(df.index):
                    oldsentid = df.get_value(index, definitions.INDEX_ID_SENTENCE)
                    i_sentence = i_sentence + 1
                dt_conll_format = dt_conll_format.append(linhanan)
                dt_conll_format = dt_conll_format.append(linha)

    # excluding auxiliar id sentence column
    dt_conll_format.drop(dt_conll_format.columns[0], axis=1, inplace=True)

    print("dataset shape: ", df.shape)
    print("done! %s sentences exported to ConLL format!", str(i_sentence))
    out_path=dataset + '.conll'
    dt_conll_format.to_csv(path_or_buf=out_path, sep="\t", index=False, header=False, float_format='%.0f')
    print("file saved to: ", out_path)

def main():
    parser = argparse.ArgumentParser(
        description='Converts a horus metadata file to CoNLL format (sentence spaces and no extra metadata information). '
                    'Especially useful for evaluating the performance of the framework in 3rd party scripts (after running the trained models)',
        prog='horus_to_conll.py',
        usage='%(prog)s [options]',
        epilog='http://horus-ner.org')

    parser.add_argument('--exp', '--experiment_folder', action='store_true', required=False,
                        help='the sub-folder name where the input file is located', default='EXP_003')
    parser.add_argument('--ds', '--input_dataset', action='store_true', required=False,
                        help='a dataset name (exported horus features)', default='ritter.horus')
    parser.add_argument('--f', '--features_ind', action='store_true', required=False,
                        help='a list contaning the indexes for each feature that should be exported',
                        default=definitions.FEATURES_INDEX)

    parser.print_help()
    args = parser.parse_args()


    try:
        convert(experiment_folder=args.exp, input_dataset_file=args.ds, features_ind=args.f)
    except:
        raise

if __name__ == "__main__":
    main()
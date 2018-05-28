import pandas as pd
import numpy as np
from src.core.util import definitions
import argparse
from src.config import HorusConfig

config = HorusConfig()

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

def convert(experiment_folder, input_dataset_file, features_ind=definitions.HORUS_FEATURES):
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
                        default=definitions.HORUS_FEATURES)

    parser.print_help()
    args = parser.parse_args()

    try:
        convert(experiment_folder=args.exp, input_dataset_file=args.ds, features_ind=args.f)
    except:
        raise

if __name__ == "__main__":
    main()
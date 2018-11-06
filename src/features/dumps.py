from src.config import HorusConfig
import multiprocessing
from src.util import definitions
from src.util.definitions import SET_MASK, dict_exp_configurations
import os
#import cPickle as pickle
import pickle
import pandas as pd
import numpy as np

config = HorusConfig()

def features_to_crf_shape(sent, i):

    features = {'bias': 1.0}
    features.update(dict((definitions.schemaindex2label[key], sent.iloc[i].at[key]) for key in np.sort(sent.columns.values)))

    if i > 0:
        features_pre = dict(('-1:' + definitions.schemaindex2label[key], sent.iloc[i-1].at[key]) for key in np.sort(sent.columns.values))
        features.update(features_pre)
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        features_pos = dict(('+1:' + definitions.schemaindex2label[key], sent.iloc[i+1].at[key]) for key in np.sort(sent.columns.values))
        features.update(features_pos)
    else:
        features['EOS'] = True

    return features


def sent2label(sent):
    return [definitions.PLOMNone_index2label[int(y)] for y in sent]

def sent2features(sent):
    return [features_to_crf_shape(sent, i) for i in range(len(sent))]

def exclude_columns(df, f_indexes):
    if isinstance(df, pd.DataFrame) == False:
        df = pd.DataFrame(df)
    #dfret = df.copy()
    out = None
    a = set(df.columns)
    b = set(f_indexes)
    out = df.drop(list(a-b), axis=1, inplace=False)

    #for icol in df.columns:
        #if icol not in f_indexes:

    return out


def mapping_indexes_text2encoding(indexes):

    import copy
    f_indexes = copy.copy(indexes)

    for i in range(len(f_indexes)):
        if f_indexes[i] == 5:
            f_indexes[i] = 91
        elif f_indexes[i] == 4:
            f_indexes[i] = 92
        elif f_indexes[i] == 85:
            f_indexes[i] = 88
        elif f_indexes[i] == 86:
            f_indexes[i] = 89
        elif f_indexes[i] == 87:
            f_indexes[i] = 90
    return f_indexes

def save_data_by_configuration((ds, dump_path, file_name, f_key, f_indexes)):

    try:

        config.logger.debug('removing columns: ' + file_name)
        f_indexes_token = mapping_indexes_text2encoding(f_indexes)
        # this is shared
        Y_sentence = [sent2label(s) for s in ds[1][1]]

        dump_path_type = dump_path.replace('.pkl', '.sentence.pkl')
        if not os.path.exists(dump_path_type):
            config.logger.debug(' -- X_sentence')
            X_sentence_str = [exclude_columns(s, f_indexes) for s in ds[1][0]]
            with open(dump_path_type, 'wb') as output1:
                pickle.dump((file_name, f_key, X_sentence_str, Y_sentence), output1, pickle.HIGHEST_PROTOCOL)
            config.logger.debug(dump_path_type + ' created!')

        dump_path_type = dump_path.replace('.pkl', '.token.pkl')
        if not os.path.exists(dump_path_type):
            config.logger.debug(' -- X_token')
            X_token = exclude_columns(ds[2][0], f_indexes_token)
            X_token.replace('O', 0, inplace=True)
            #Y_token = [definitions.PLOMNone_label2index[y] for y in ds[2][1]]
            Y_token = [int(y) for y in ds[2][1]]

            with open(dump_path_type, 'wb') as output2:
                pickle.dump((file_name, f_key, X_token, Y_token), output2, pickle.HIGHEST_PROTOCOL)
            config.logger.debug(dump_path_type + ' created!')

        dump_path_type = dump_path.replace('.pkl', '.crf.pkl')
        if not os.path.exists(dump_path_type):
            config.logger.debug(' -- X_crf')
            X_crf = [sent2features(s) for s in X_sentence_str]
            # trick for scikit-learn on CRF (for the precision_recall_f-score_support method)
            Y_crf = np.array([x for s in Y_sentence for x in s])
            with open(dump_path_type, 'wb') as output3:
                pickle.dump((file_name, f_key, X_crf, Y_crf), output3, pickle.HIGHEST_PROTOCOL)
            config.logger.debug(dump_path_type + ' created!')

        dump_path_type = dump_path.replace('.pkl', '.sentence.idx.pkl')
        if not os.path.exists(dump_path_type):
            config.logger.debug(' -- X_sentence_encoded')
            X_sentence_idx = [exclude_columns(s, f_indexes_token) for s in ds[1][0]]
            with open(dump_path_type, 'wb') as output2:
                pickle.dump((file_name, f_key, X_sentence_idx, Y_sentence), output2, pickle.HIGHEST_PROTOCOL)
            config.logger.debug(dump_path_type + ' created!')

        ## X_lstm, y_lstm, max_features, out_size, maxlen = convert_lstm_shape(X_sentence, Y_sentence, f_indexes)
        ## X2_lstm, y2_lstm, max_features_2, out_size_2, maxlen_2 = convert_lstm_shape(ds2[1][0], ds2[1][1], f_indexes)
        ## X1_lstm = pad_sequences(X1_lstm, maxlen=max(maxlen_1, maxlen_2))
        ## y1_lstm = pad_sequences(y1_lstm, maxlen=max(maxlen_1, maxlen_2))

        #with open(dump_path, 'wb') as output:
        #    pickle.dump((file_name, f_key, X_sentence, Y_sentence, X_token, Y_token, X_crf, _Y_sentence),
        #                output, pickle.HIGHEST_PROTOCOL)
        #config.logger.debug(dump_path + ' created!')

    except Exception as e:
        config.logger.error(repr(e))
        raise e


def create_benchmark_dump_files():
    try:
        job_dumps = []
        for ds in definitions.NER_DATASETS:
            horus_m4_path = ds[1] + ds[2].replace('.horusx', '.horus4')
            horus_m4_name = ds[0]
            if not os.path.isfile(horus_m4_path):
                config.logger.error(
                    ' -- file .horus4 does not exist! please check the file extract_lex.py to create it...')
            else:
                data = None
                for key, value in dict_exp_configurations.items():
                    dump_name = SET_MASK % (horus_m4_name, str(key))
                    dump_full_path = os.path.dirname(os.path.realpath(horus_m4_path)) + '/' +  dump_name
                    # this may lead to error, but I am considering pre-processing worked fine for now.
                    if not os.path.exists(dump_full_path.replace('.pkl', '.sentence.idx.pkl')):
                        config.logger.debug(' -- key: ' + str(key))
                        if data is None:
                            config.logger.debug('loading: ' + horus_m4_path)
                            with open(horus_m4_path, 'rb') as input:
                                data = pickle.load(input)
                        job_dumps.append((data, dump_full_path, dump_name, key, value))

            if len(job_dumps) > 0:
                config.logger.info('creating dump files: ' + str(len(job_dumps)) + ' jobs')
                p = multiprocessing.Pool(multiprocessing.cpu_count())
                p.map(save_data_by_configuration, job_dumps)
                config.logger.info('dump files generated successfully')

    except Exception as e:
        config.logger.error(repr(e))


def main():

    try:
        create_benchmark_dump_files()
    except:
        raise

if __name__ == "__main__":
    main()
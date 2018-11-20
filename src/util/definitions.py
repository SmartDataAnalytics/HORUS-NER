import os

import sklearn_crfsuite
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV

from src.config import HorusConfig

config = HorusConfig()
if config.root_dir == '':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    ROOT_DIR = config.root_dir

RUN_TAGGER_CMD = config.models_tweetnlp_java_param + " -jar " + config.models_tweetnlp_jar + " --model " + config.models_tweetnlp_model

SET_MASK = '_%s_config_%s.pkl'


NER_BROAD_PER = ['B-PER', 'I-PER']
NER_BROAD_LOC = ['B-LOC', 'I-LOC']
NER_BROAD_ORG = ['B-ORG', 'I-ORG']

NER_WNUT_PER = ['B-person', 'I-person', 'person']
NER_WNUT_LOC = ['B-location', 'I-location', 'location']
NER_WNUT_ORG = ['B-corporation', 'I-corporation', 'corporation']

NER_RITTER_PER = ['B-person', 'I-person']
NER_RITTER_ORG = ['B-company', 'I-company']
NER_RITTER_LOC = ['B-geo-loc', 'I-geo-loc']

NER_STANFORD_PER = ['PERSON']
NER_STANFORD_ORG = ['ORGANIZATION', 'GSP'] # GSP = geo-political social group
NER_STANFORD_LOC = ['LOCATION']

NER_NLTK_PER = ['B-PERSON', 'I-PERSON', 'PERSON']
NER_NLTK_ORG = ['B-ORGANIZATION', 'I-ORGANIZATION', 'ORGANIZATION', 'GSP']
NER_NLTK_LOC = ['B-LOCATION', 'I-LOCATION', 'LOCATION', 'GPE'] # GPE = geo-political entities such as city, state/province, and country

NER_CONLL_PER = ['I-PER']
NER_CONLL_ORG = ['I-ORG']
NER_CONLL_LOC = ['I-LOC'] # GPE = geo-political entities such as city, state/province, and country

NER_TAGS_MISC = ['MISC', 'other', 'B-other', 'I-other', 'B-misc', 'I-misc']

NER_TAGS_PER = ['PER']
NER_TAGS_PER.extend(NER_RITTER_PER)
NER_TAGS_PER.extend(NER_STANFORD_PER)
NER_TAGS_PER.extend(NER_NLTK_PER)
NER_TAGS_PER.extend(NER_CONLL_PER)
NER_TAGS_PER.extend(NER_WNUT_PER)
NER_TAGS_PER.extend(NER_BROAD_PER)

NER_TAGS_ORG = ['ORG']
NER_TAGS_ORG.extend(NER_RITTER_ORG)
NER_TAGS_ORG.extend(NER_STANFORD_ORG)
NER_TAGS_ORG.extend(NER_NLTK_ORG)
NER_TAGS_ORG.extend(NER_CONLL_ORG)
NER_TAGS_ORG.extend(NER_WNUT_ORG)
NER_TAGS_ORG.extend(NER_BROAD_ORG)

NER_TAGS_LOC = ['LOC']
NER_TAGS_LOC.extend(NER_RITTER_LOC)
NER_TAGS_LOC.extend(NER_STANFORD_LOC)
NER_TAGS_LOC.extend(NER_NLTK_LOC)
NER_TAGS_LOC.extend(NER_CONLL_LOC)
NER_TAGS_LOC.extend(NER_WNUT_LOC)
NER_TAGS_LOC.extend(NER_BROAD_LOC)

NER_TAGS = []
NER_TAGS.extend(NER_TAGS_LOC)
NER_TAGS.extend(NER_TAGS_ORG)
NER_TAGS.extend(NER_TAGS_PER)


encoder_int_words_name = 'encoder_int_ritterwnut151617_word.pkl'
encoder_int_lemma_name ='encoder_int_ritterwnut151617_lemma.pkl'
encoder_int_stem_name = 'encoder_int_ritterwnut151617_stem.pkl'
encoder_onehot_words_name = 'encoder_onehot_ritterwnut151617_word.pkl'
encoder_onehot_lemma_name ='encoder_onthot_ritterwnut151617_lemma.pkl'
encoder_onehot_stem_name = 'encoder_onehot_ritterwnut151617_stem.pkl'
encoder_le1_name='_encoder_pos.pkl'
encoder_le2_name='_encoder_nltk2.pkl'

# Penn Treebank
POS_NOUN_PTB = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$']
POS_NOUN_UNIVERSAL = ['NOUN', 'PRON', 'PROPN']

POS_NOUN_TAGS = []
POS_NOUN_TAGS.extend(POS_NOUN_PTB)
POS_NOUN_TAGS.extend(POS_NOUN_UNIVERSAL)

NER_RITTER = []
NER_RITTER.extend(NER_RITTER_PER)
NER_RITTER.extend(NER_RITTER_ORG)
NER_RITTER.extend(NER_RITTER_LOC)

NER_CONLL = []
NER_CONLL.extend(NER_CONLL_PER)
NER_CONLL.extend(NER_CONLL_ORG)
NER_CONLL.extend(NER_CONLL_LOC)

NER_TAGS = []
NER_TAGS.extend(NER_TAGS_ORG)
NER_TAGS.extend(NER_TAGS_PER)
NER_TAGS.extend(NER_TAGS_LOC)
NER_TAGS.extend(NER_TAGS_MISC)

# PER, LOC, ORG and MISC
PLOMNone_index2label = {1: "LOC", 2: "ORG", 3: "PER", 4: "MISC", 5: "O"} #KLASSES
PLOMNone_label2index = {"LOC": 1, "ORG": 2, "PER": 3, "MISC": 4, "O": 5} #KLASSES2
PLOM_index2label = PLOMNone_index2label.copy()
del PLOM_index2label[5]
# not testing MISC for now
del PLOM_index2label[4]

header = 'cross-validation\tconfig\trun\tlabel\tprecision\trecall\tf1\tsupport\talgo\tdataset1\tdataset2\ttask\n'
line = '%s\t%s\t%s\t%s\t%.5f\t%.5f\t%.5f\t%s\t%s\t%s\t%s\t%s\n'

def tags_to_3muc_simple(tags):
    for i in range(len(tags)):
        if tags[i] in NER_TAGS_PER:
            tags[i] = PLOMNone_label2index['PER']
        elif tags[i] in NER_TAGS_ORG:
            tags[i] = PLOMNone_label2index['ORG']
        elif tags[i] in NER_TAGS_LOC:
            tags[i] = PLOMNone_label2index['LOC']
        else:
            tags[i] = PLOMNone_label2index['O']
    return tags



HORUS_MATRIX_HEADER = ["IS_NAMED_ENTITY", "ID_SENT", "ID_WORD", "TOKEN", "POS_UNI", "POS", "NER", "COMPOUND",
    "COMPOUND_SIZE", "ID_TERM_TXT", "ID_TERM_IMG", "TOT_IMG", "TOT_CV_LOC", "TOT_CV_ORG",
    "TOT_CV_PER", "DIST_CV_I", "PL_CV_I", "NR_RESULTS_SE_IMG", "KLASS_PREDICT_CV", "TOT_RESULTS_TX", "TOT_TX_LOC",
    "TOT_TX_ORG", "TOT_TX_PER", "TOT_ERR_TRANS", "DIST_TX_I", "NR_RESULTS_SE_TX", "KLASS_PREDICT_TX",
    "FEATURE_EXTRA_1", "FEATURE_EXTRA_2", "FEATURE_EXTRA_3", "FEATURE_EXTRA_4", "FEATURE_EXTRA_5", "FEATURE_EXTRA_6",
    "FEATURE_EXTRA_7", "FEATURE_EXTRA_8", "FEATURE_EXTRA_9", "KLASS_1", "KLASS_2", "KLASS_3", "KLASS_4",
    "KLASS_5", "KLASS_6", "KLASS_7", "KLASS_8", "KLASS_9", "KLASS_10", "KLASS_11", "KLASS_12", "KLASS_13",
    "KLASS_14", "KLASS_15", "KLASS_REAL"]

CMU_PENN_TAGS = [['N', 'NNS'], ['O', 'PRP'], ['S', 'PRP$'], ['^', 'NNP'], ["D", "DT"], ["A", "JJ"], ["P", "IN"],
                     ["&", "CC"],["V", "VBD"], ["R", "RB"], ["!", "UH"], ["T", "RP"], ["$", "CD"], ['G', 'SYM']]

CMU_UNI_TAGS = [["N", "NOUN"], ["^", "NOUN"], ["V", "VERB"], ["D", "DET"], ["A", "ADJ"], ["P", "ADP"],
                        ["&", "CCONJ"], ["R", "ADV"], ["!", "INTJ"], ["O","PRON"], ["$", "NUM"], [",", "PUNCT"]]

PENN_UNI_TAG = [['#', 'SYM'],['$', 'SYM'], ['','PUNCT'],[',','PUNCT'],['-LRB-','PUNCT'],['-RRB-','PUNCT'],['.','PUNCT'],[':','PUNCT'],	['AFX','ADJ'],
                    ['CC','CONJ'],['CD','NUM'],['DT','DET'],['EX','ADV'],['FW','X'],['HYPH','PUNCT'],['IN','ADP'],['JJ','ADJ'],	['JJR','ADJ'],['JJS','ADJ'],
                    ['LS','PUNCT'],['MD','VERB'],['NIL','X'],['NN','NOUN'],	['NNP','PROPN'],['NNPS','PROPN'],['NNS','NOUN'],['PDT','DET'],['POS','PART'],
                    ['PRP','PRON'],['PRP$','DET'],['RB','ADV'],['RBR','ADV'],['RBS','ADV'],['RP','PART'],['SYM','SYM'],['TO','PART'],['UH','INTJ'],['VB','VERB'],
                    ['VBD','VERB'],['VBG','VERB'],['VBN','VERB'],['VBP','VERB'],['VBZ','VERB'],['WDT','DET'],['WP','PRON'],['WP$', 'DET'],['WRB', 'ADV']]


seeds_dict_img_classes = {'per': ['person', 'human being', 'man', 'woman', 'child', 'human body', 'human face'],
                      'loc': ['location', 'place', 'residence', 'landscape', 'building', 'volcano', 'stone', 'country', 'beach', 'sky', 'road', 'ocean', 'sea', 'lake', 'square', 'map', 'flag', 'city', 'forest'],
                      'org': ['organisation', 'logo', 'logotype'],
                     'none': ['clipper', 'animal', 'telephone', 'time', 'cup', 'table', 'bottle', 'window', 'vehicle' 'monitor']}

seeds_dict_topics = {'per': ['guy', 'girl', 'woman', 'husband', 'baby', 'people', 'human', 'person', 'man', 'child',
                             'celebrity','arnett', 'david', 'richard', 'james', 'frank', 'george', 'misha',
                'student', 'education', 'coach', 'football', 'turkish', 'actor',
                'albanian', 'romanian', 'professor', 'lawyer', 'president',
                'king', 'danish', 'we', 'he', 'their', 'born',
                'directed', 'died', 'lives', 'boss', 'syrian', 'elected',
                'minister', 'candidate', 'daniel', 'robert', 'dude'],
        'loc': ['landscape', 'country', 'location', 'place', 'building', 'highway', 'forest', 'sea', 'mountain', 'city',
                'china', 'usa', 'germany', 'leipzig', 'alaska', 'poland',
                'jakarta', 'kitchen', 'house', 'brazil', 'fuji', 'prison',
                'portugal', 'lisbon', 'france', 'oslo', 'airport', 'road',
                'stadium', 'hospital', 'temple', 'beach', 'hotel', 'state', 'home',
                'world', 'island', 'land' ,'waterfall', 'kitchen', 'room', 'office',
                'bedroom', 'bathroom', 'hall', 'castle', 'flag', 'map'],
        'org': ['office', 'startup', 'organisation' 'enterprise', 'venture', 'company', 'business', 'industry', 'headquarters', 'foundation',
                'microsoft', 'bloomberg', 'google', 'contract', 'project', 'research',  'capital', 'milestones', 'risk',
                'funded', 'idea', 'product', 'client', 'investment', 'certification', 'news', 'logo', 'trademark', 'job'],
        'none': ['frog', 'animal', 'monkey', 'dog', 'skate', 'cup', 'money', 'cash',
                 'mouse', 'snake', 'telephone', 'glass', 'monitor', 'bible', 'book',
                 'dictionary', 'religion', 'politics', 'sports', 'question', 'linux',
                 'java', 'python', 'months', 'time', 'wallet', 'umbrella', 'cable',
                 'internet', 'connection', 'pencil', 'earphone', 'shopping', 'buy',
                 'headphones', 'bread', 'food', 'cake', 'bottle', 'table', 'jacket',
                 'politics', 'computer', 'laptop', 'blue', 'green', 'bucket', 'orange', 'rose',
                 'key', 'clock', 'connector']}




# basic info
INDEX_IS_ENTITY = 0
INDEX_ID_SENTENCE = 1
INDEX_ID_WORD = 2
INDEX_TOKEN = 3
INDEX_POS_UNI = 4
INDEX_POS = 5
INDEX_NER = 6
INDEX_IS_COMPOUND = 7
INDEX_COMPOUND_SIZE = 8
INDEX_ID_TERM_TXT = 9
INDEX_ID_TERM_IMG = 10

# basic CV
INDEX_TOT_IMG = 11
INDEX_TOT_CV_LOC = 12
INDEX_TOT_CV_ORG = 13
INDEX_TOT_CV_PER = 14
INDEX_DIST_CV_I = 15
INDEX_PL_CV_I= 16
INDEX_NR_RESULTS_SE_IMG = 17
INDEX_MAX_KLASS_PREDICT_CV = 18

# basic TX
INDEX_TOT_RESULTS_TX = 19
INDEX_TOT_TX_LOC = 20
INDEX_TOT_TX_ORG = 21
INDEX_TOT_TX_PER = 22
INDEX_TOT_ERR_TRANS = 23
INDEX_DIST_TX_I = 24
INDEX_NR_RESULTS_SE_TX = 25
INDEX_MAX_KLASS_PREDICT_TX = 26

# cnn TX
INDEX_TOT_TX_LOC_TM_CNN = 28
INDEX_TOT_TX_ORG_TM_CNN = 29
INDEX_TOT_TX_PER_TM_CNN = 30
INDEX_DIST_TX_I_TM_CNN = 31
INDEX_TOT_EMB_SIMILAR_LOC = 42
INDEX_TOT_EMB_SIMILAR_ORG = 43
INDEX_TOT_EMB_SIMILAR_PER = 44
INDEX_TOT_EMB_SIMILAR_NONE = 49
INDEX_TOT_TX_NONE_TM_CNN = 50

# schema
schemaindex2label = {
    0: 'is_entity',
    1: 'index_sentence',
    2: 'index_word',
    3: 'token',
    4: 'pos_universal',
    5: 'pos',
    6: 'ner',
    7: 'is_compound',
    8: 'compound_size',
    9: 'id_term_txt',
    10: 'id_term_img',
    11: 'tot_results_img',
    12: 'tot_cv_loc',
    13: 'tot_cv_org',
    14: 'tot_cv_per',
    15: 'dist_cv_i',
    16: 'pl_cv_i',
    17: 'tot_results_se_img',
    18: 'max_klass_predicted_cv',
    19: 'tot_results_tx',
    20: 'tot_tx_loc',
    21: 'tot_tx_org',
    22: 'tot_tx_per',
    23: 'tot_err_trans',
    24: 'dist_tx_i',
    25: 'tot_results_se_txt',
    26: 'max_klass_predicted_tx',
    27: '',
    28: 'tot_tx_loc_tm_cnn',
    29: 'tot_tx_org_tm_cnn',
    30: 'tot_tx_per_tm_cnn',
    31: 'dist_tc_i_tm_cnn',
    32: 'tot_cv_loc_1_cnn',
    33: 'tot_cv_org_cnn',
    34: 'tot_cv_per_cnn',
    35: 'tot_cv_loc_2_cnn',
    36: 'tot_cv_loc_3_cnn',
    37: 'tot_cv_loc_4_cnn',
    38: '',
    39: 'max_klass_predict_compound',
    40: 'klass_final_model',
    41: 'max_klass_predict_cv_cnn',
    42: 'tot_emb_similar_loc',
    43: 'tot_emb_similar_org',
    44: 'tot_emb_similar_per',
    45: 'tot_cv_loc_5_cnn',
    46: 'tot_cv_loc_6_cnn',
    47: 'tot_cv_loc_7_cnn',
    48: 'tot_cv_loc_8_cnn',
    49: 'tot_emb_similar_none',
    50: 'tot_tx_none_tm_cnn',
    51: 'y',
    52: 'tx_cnn_stat_sum_loc',
    53: 'tx_cnn_stat_sum_org',
    54: 'tx_cnn_stat_sum_per',
    55: 'tx_cnn_stat_sum_none',
    56: 'tx_cnn_stat_avg_loc',
    57: 'tx_cnn_stat_avg_org',
    58: 'tx_cnn_stat_avg_per',
    59: 'tx_cnn_stat_avg_none',
    60: 'tx_cnn_stat_max_loc',
    61: 'tx_cnn_stat_max_org',
    62: 'tx_cnn_stat_max_per',
    63: 'tx_cnn_stat_max_none',
    64: 'tx_cnn_stat_min_loc',
    65: 'tx_cnn_stat_min_org',
    66: 'tx_cnn_stat_min_per',
    67: 'tx_cnn_stat_min_none',
    68: 'tx_cnn_stat_t_plus_top5_k_sum_loc',
    69: 'tx_cnn_stat_t_plus_top5_k_sum_org',
    70: 'tx_cnn_stat_t_plus_top5_k_sum_per',
    71: 'tx_cnn_stat_t_plus_top5_k_sum_none',
    72: 'tx_cnn_stat_t_plus_top5_k_avg_loc',
    73: 'tx_cnn_stat_t_plus_top5_k_avg_org',
    74: 'tx_cnn_stat_t_plus_top5_k_avg_per',
    75: 'tx_cnn_stat_t_plus_top5_k_avg_none',
    76: 'tx_cnn_stat_t_plus_top5_k_max_loc',
    77: 'tx_cnn_stat_t_plus_top5_k_max_org',
    78: 'tx_cnn_stat_t_plus_top5_k_max_per',
    79: 'tx_cnn_stat_t_plus_top5_k_max_none',
    80: 'tx_cnn_stat_t_plus_top5_k_min_loc',
    81: 'tx_cnn_stat_t_plus_top5_k_min_org',
    82: 'tx_cnn_stat_t_plus_top5_k_min_per',
    83: 'tx_cnn_stat_t_plus_top5_k_min_none',
    84: '',
    85: 'word.lower',
    86: 'word.lemma',
    87: 'word.stem',
    88: 'word.lower.encoded',
    89: 'word.lemma.encoded',
    90: 'word.stem.encoded',
    91: 'pos.encoded',
    92: 'pos_universal.encoded',
    93: 'word.len.1',
    94: 'word.has.url',
    95: 'word[0].isupper',
    96: 'word.isupper',
    97: 'word.istitle',
    98: 'word.isdigit',
    99: 'word.stop',
    100: 'word.len.issmall',
    101: 'word.has.minus',
    102: 'word.shape',
    103: 'brown_320.1',
    104: 'brown_320.2',
    105: 'brown_320.3',
    106: 'brown_320.4',
    107: 'brown_320.5',
    108: 'brown_640.1',
    109: 'brown_640.2',
    110: 'brown_640.3',
    111: 'brown_640.4',
    112: 'brown_640.5',
    113: 'brown_1000.1',
    114: 'brown_1000.2',
    115: 'brown_1000.3',
    116: 'brown_1000.4',
    117: 'brown_1000.5'
}



#pos, pos_uni, len==1, tot_special, word[0]==upper, word==upper, title, digit, stop, len<2, -, shape

# stats cnn TX
INDEX_TX_CNN_STAT_SUM_LOC = 52
INDEX_TX_CNN_STAT_SUM_ORG = 53
INDEX_TX_CNN_STAT_SUM_PER = 54
INDEX_TX_CNN_STAT_SUM_NONE = 55
INDEX_TX_CNN_STAT_AVG_LOC = 56
INDEX_TX_CNN_STAT_AVG_ORG = 57
INDEX_TX_CNN_STAT_AVG_PER = 58
INDEX_TX_CNN_STAT_AVG_NONE = 59
INDEX_TX_CNN_STAT_MAX_LOC = 60
INDEX_TX_CNN_STAT_MAX_ORG = 61
INDEX_TX_CNN_STAT_MAX_PER = 62
INDEX_TX_CNN_STAT_MAX_NONE = 63
INDEX_TX_CNN_STAT_MIN_LOC = 64
INDEX_TX_CNN_STAT_MIN_ORG = 65
INDEX_TX_CNN_STAT_MIN_PER = 66
INDEX_TX_CNN_STAT_MIN_NONE = 67
INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_SUM_LOC = 68
INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_SUM_ORG = 69
INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_SUM_PER = 70
INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_SUM_NONE = 71
INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_AVG_LOC = 72
INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_AVG_ORG = 73
INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_AVG_PER = 74
INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_AVG_NONE = 75
INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MAX_LOC = 76
INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MAX_ORG = 77
INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MAX_PER = 78
INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MAX_NONE = 79
INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MIN_LOC = 80
INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MIN_ORG = 81
INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MIN_PER = 82
INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MIN_NONE = 83

# cnn CV
INDEX_TOT_CV_LOC_1_CNN = 32
INDEX_TOT_CV_ORG_CNN = 33
INDEX_TOT_CV_PER_CNN = 34
INDEX_TOT_CV_LOC_2_CNN = 35
INDEX_TOT_CV_LOC_3_CNN = 36
INDEX_TOT_CV_LOC_4_CNN = 37
#INDEX_MAX_KLASS_PREDICT_TX_CNN = 38
INDEX_TOT_CV_LOC_5_CNN = 45
INDEX_TOT_CV_LOC_6_CNN = 46
INDEX_TOT_CV_LOC_7_CNN = 47
INDEX_TOT_CV_LOC_8_CNN = 48


INDEX_MAX_KLASS_PREDICT_COMPOUND = 39
INDEX_KLASS_FINAL_MODEL = 40
INDEX_MAX_KLASS_PREDICT_CV_CNN = 41

INDEX_TARGET_NER = 51

HORUS_TOT_FEATURES = 84


STANDARD_FEAT_WORD_LEN = 3 # word, lemma, stem
STANDARD_FEAT_BASIC_LEN = 12 #pos, pos_uni, len==1, tot_special, word[0]==upper, word==upper, title, digit, stop, len<2, -, shape
STANDARD_FEAT_BROWN_LEN= 15 # brown320 (5), brown640 (5), brown1000 (5)
#----------------------------
STANDARD_FEAT_LEN = STANDARD_FEAT_WORD_LEN + STANDARD_FEAT_BASIC_LEN + STANDARD_FEAT_BROWN_LEN
'''
STANDARD FEATURES
'''

# label, path, file
import numpy as np
trees_param_bootstrap = {"max_features": ['auto', 'sqrt'],
                     "max_depth": [int(x) for x in np.linspace(10, 110, num=11)],
                     "min_samples_split": [2, 5, 10],
                     "min_samples_leaf": [1, 2, 4],
                     "n_estimators": [10, 25, 50, 100, 200, 400, 600, 800],
                     "bootstrap": [True, False]
}

import scipy
crf_param = {'c1': scipy.stats.expon(scale=0.5),
         'c2': scipy.stats.expon(scale=0.05),
        'algorithm': ['lbfgs', 'pa'],
}

optim_clf_rf = RandomizedSearchCV(RandomForestClassifier(),
                                  trees_param_bootstrap,
                                  verbose=1,
                                  cv=5, scoring=['precision', 'recall', 'f1'],
                                  n_jobs=-1, refit='f1', n_iter=20)
optim_clf_crf = RandomizedSearchCV(sklearn_crfsuite.CRF(all_possible_transitions=True, max_iterations=100),
                        crf_param,
                         verbose=1,
                         cv=5, scoring=['precision', 'recall', 'f1'],
                         n_jobs=-1, refit='f1', n_iter=20)

NER_DATASETS = [
            ['ritter.train', config.dir_datasets + 'Ritter/', 'ner.txt.horusx'],
            ['wnut15.train', config.dir_datasets + 'wnut/2015/data/', 'train.horusx'],
            ['wnut15.dev',   config.dir_datasets + 'wnut/2015/data/', 'dev.horusx'],
            ['wnut16.train', config.dir_datasets + 'wnut/2016/data/', 'train.horusx'],
            ['wnut16.dev',   config.dir_datasets + 'wnut/2016/data/', 'dev.horusx'],
            ['wnut16.test',  config.dir_datasets + 'wnut/2016/data/', 'test.horusx'],
            ['wnut17.train', config.dir_datasets + 'wnut/2017/', 'wnut17train.conll.horusx'],
            ['wnut17.dev',   config.dir_datasets + 'wnut/2017/', 'emerging.dev.conll.horusx'],
            ['wnut17.test',  config.dir_datasets + 'wnut/2017/', 'emerging.test.annotated.horusx']
        ]


# [[train], [dev]]
NER_DATASETS_TRAIN_DEV = [
            [['ritter.train', config.dir_datasets + 'Ritter/', 'ner.txt.horusx'],
             [None, None, None]],
            [['wnut15.train', config.dir_datasets + 'wnut/2015/data/', 'train.horusx'],
             [None, None, None]],
            [['wnut16.train', config.dir_datasets + 'wnut/2016/data/', 'train.horusx'],
             ['wnut16.dev', config.dir_datasets + 'wnut/2016/data/', 'dev.horusx']],
            [['wnut17.train', config.dir_datasets + 'wnut/2017/', 'wnut17train.conll.horusx'],
             ['wnut17.dev', config.dir_datasets + 'wnut/2017/', 'emerging.dev.conll.horusx']]
        ]

NER_DATASETS_TEST = [
            ['wnut15.dev',   config.dir_datasets + 'wnut/2015/data/', 'dev.horusx'], #wnut15.dev ==wnut test set
            ['wnut16.test',  config.dir_datasets + 'wnut/2016/data/', 'test.horusx'],
            ['wnut17.test',  config.dir_datasets + 'wnut/2017/', 'emerging.test.annotated.horusx']
        ]



# 'broad/5d7c65d/a.conll', 'broad/5d7c65d/b.conll', 'broad/5d7c65d/e.conll', 'broad/5d7c65d/f.conll',
#              'broad/5d7c65d/g.conll', 'broad/5d7c65d/h.conll'

EXPERIMENT_FOLDER = 'EXP005/'


# attention: not encoded to speed up data generation. do the encoding at benchmark level
FEATURES_STANDARD = [85, 4, 5, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102]
FEATURES_LEMMA = [86, 87]
FEATURES_BROWN_64M_c320 = [103, 104, 105, 106, 107]
FEATURES_BROWN_64M_c640 = [108, 109, 110, 111, 112]
FEATURES_BROWN_500M_c1000 = [113, 114, 115, 116, 117]



FEATURES_STANDARD_BROWN_BEST = FEATURES_STANDARD + FEATURES_BROWN_64M_c320
'''
HORUS FEATURES
'''
FEATURES_HORUS_BASIC_TX      = [INDEX_TOT_RESULTS_TX, INDEX_TOT_TX_LOC, INDEX_TOT_TX_ORG, INDEX_TOT_TX_PER, INDEX_TOT_ERR_TRANS, INDEX_DIST_TX_I, INDEX_NR_RESULTS_SE_TX]
FEATURES_HORUS_CNN_TX        = [INDEX_TOT_TX_LOC_TM_CNN, INDEX_TOT_TX_ORG_TM_CNN, INDEX_TOT_TX_PER_TM_CNN, INDEX_DIST_TX_I_TM_CNN, INDEX_TOT_TX_NONE_TM_CNN]
FEATURES_HORUS_BASIC_CV      = [INDEX_TOT_IMG, INDEX_TOT_CV_LOC, INDEX_TOT_CV_ORG, INDEX_TOT_CV_PER, INDEX_DIST_CV_I, INDEX_PL_CV_I, INDEX_NR_RESULTS_SE_IMG]
FEATURES_HORUS_CNN_CV        = [INDEX_TOT_CV_LOC_1_CNN, INDEX_TOT_CV_LOC_2_CNN, INDEX_TOT_CV_LOC_3_CNN, INDEX_TOT_CV_LOC_4_CNN, INDEX_TOT_CV_LOC_5_CNN, INDEX_TOT_CV_LOC_6_CNN, INDEX_TOT_CV_LOC_7_CNN, INDEX_TOT_CV_LOC_8_CNN, INDEX_TOT_CV_ORG_CNN, INDEX_TOT_CV_PER_CNN]
FEATURES_HORUS_EMB_TX        = [INDEX_TOT_EMB_SIMILAR_LOC, INDEX_TOT_EMB_SIMILAR_ORG, INDEX_TOT_EMB_SIMILAR_PER, INDEX_TOT_EMB_SIMILAR_NONE]
FEATURES_HORUS_STATS_TX      = [INDEX_TX_CNN_STAT_SUM_LOC, INDEX_TX_CNN_STAT_SUM_ORG, INDEX_TX_CNN_STAT_SUM_PER, INDEX_TX_CNN_STAT_SUM_NONE, INDEX_TX_CNN_STAT_AVG_LOC, INDEX_TX_CNN_STAT_AVG_ORG, INDEX_TX_CNN_STAT_AVG_PER, INDEX_TX_CNN_STAT_AVG_NONE, INDEX_TX_CNN_STAT_MAX_LOC, INDEX_TX_CNN_STAT_MAX_ORG, INDEX_TX_CNN_STAT_MAX_PER, INDEX_TX_CNN_STAT_MAX_NONE, INDEX_TX_CNN_STAT_MIN_LOC, INDEX_TX_CNN_STAT_MIN_ORG, INDEX_TX_CNN_STAT_MIN_PER, INDEX_TX_CNN_STAT_MIN_NONE, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_SUM_LOC, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_SUM_ORG, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_SUM_PER, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_SUM_NONE, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_AVG_LOC, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_AVG_ORG, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_AVG_PER, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_AVG_NONE, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MAX_LOC, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MAX_ORG, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MAX_PER, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MAX_NONE, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MIN_LOC, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MIN_ORG, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MIN_PER, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MIN_NONE]
FEATURES_HORUS_TX_FULL       = FEATURES_HORUS_BASIC_TX + FEATURES_HORUS_CNN_TX
FEATURES_HORUS_CV_FULL       = FEATURES_HORUS_BASIC_CV + FEATURES_HORUS_CNN_CV
FEATURES_HORUS_BASIC_AND_CV_CNN = FEATURES_HORUS_BASIC_TX + FEATURES_HORUS_BASIC_CV + FEATURES_HORUS_CNN_CV
FEATURES_HORUS_BASIC_AND_TX_CNN = FEATURES_HORUS_BASIC_TX + FEATURES_HORUS_BASIC_CV + FEATURES_HORUS_CNN_TX


dict_exp_configurations = {1:  FEATURES_STANDARD,
                           2:  list(set().union(FEATURES_STANDARD, FEATURES_HORUS_BASIC_TX)),
                           3:  list(set().union(FEATURES_STANDARD, FEATURES_HORUS_BASIC_CV)),
                           4:  list(set().union(FEATURES_STANDARD, FEATURES_HORUS_BASIC_TX, FEATURES_HORUS_BASIC_CV)),
                           5:  list(set().union(FEATURES_STANDARD, FEATURES_LEMMA)),
                           6:  list(set().union(FEATURES_STANDARD, FEATURES_LEMMA, FEATURES_HORUS_BASIC_TX)),  #igual a 2
                           7:  list(set().union(FEATURES_STANDARD, FEATURES_LEMMA, FEATURES_HORUS_BASIC_CV)),  #melhor que 3
                           8:  list(set().union(FEATURES_STANDARD, FEATURES_LEMMA, FEATURES_HORUS_BASIC_TX, FEATURES_HORUS_BASIC_CV)),  #nao eh melhor que 4
                           9:  list(set().union(FEATURES_STANDARD, FEATURES_BROWN_64M_c320)),
                           10: list(set().union(FEATURES_STANDARD, FEATURES_BROWN_64M_c640)),
                           11: list(set().union(FEATURES_STANDARD, FEATURES_BROWN_500M_c1000)),
                           12: list(set().union(FEATURES_STANDARD, FEATURES_LEMMA, FEATURES_BROWN_64M_c320)),
                           13: list(set().union(FEATURES_STANDARD, FEATURES_LEMMA, FEATURES_BROWN_64M_c640)),
                           14: list(set().union(FEATURES_STANDARD, FEATURES_LEMMA, FEATURES_BROWN_500M_c1000)),
                           15: list(set().union(FEATURES_STANDARD_BROWN_BEST, FEATURES_HORUS_BASIC_CV)),
                           16: list(set().union(FEATURES_STANDARD_BROWN_BEST, FEATURES_HORUS_BASIC_TX)),
                           17: list(set().union(FEATURES_STANDARD_BROWN_BEST, FEATURES_HORUS_BASIC_CV, FEATURES_HORUS_BASIC_TX)),
                           18: list(set().union(FEATURES_STANDARD, FEATURES_HORUS_CNN_CV)),
                           19: list(set().union(FEATURES_STANDARD, FEATURES_HORUS_CNN_TX)),
                           20: list(set().union(FEATURES_STANDARD, FEATURES_HORUS_EMB_TX)),
                           21: list(set().union(FEATURES_STANDARD, FEATURES_HORUS_STATS_TX)),
                           22: list(set().union(FEATURES_STANDARD, FEATURES_HORUS_TX_FULL)),
                           23: list(set().union(FEATURES_STANDARD, FEATURES_HORUS_TX_FULL, FEATURES_HORUS_EMB_TX, FEATURES_HORUS_STATS_TX)),
                           24: list(set().union(FEATURES_STANDARD, FEATURES_HORUS_CNN_CV + FEATURES_HORUS_CNN_TX)),
                           25: list(set().union(FEATURES_STANDARD, FEATURES_HORUS_BASIC_AND_TX_CNN)),
                           26: list(set().union(FEATURES_STANDARD, FEATURES_HORUS_CV_FULL)),
                           27: list(set().union(FEATURES_STANDARD, FEATURES_HORUS_BASIC_AND_CV_CNN)),
                           28: list(set().union(FEATURES_STANDARD, FEATURES_HORUS_TX_FULL, FEATURES_HORUS_CV_FULL)),
                           29: list(set().union(FEATURES_STANDARD, FEATURES_HORUS_TX_FULL, FEATURES_HORUS_CV_FULL, FEATURES_HORUS_EMB_TX,
                                                FEATURES_HORUS_STATS_TX)),
                           30: list(set().union(FEATURES_STANDARD_BROWN_BEST, FEATURES_HORUS_CNN_CV)),
                           31: list(set().union(FEATURES_STANDARD_BROWN_BEST, FEATURES_HORUS_CNN_TX)),
                           32: list(set().union(FEATURES_STANDARD_BROWN_BEST, FEATURES_HORUS_EMB_TX)),
                           33: list(set().union(FEATURES_STANDARD_BROWN_BEST, FEATURES_HORUS_STATS_TX)),
                           34: list(set().union(FEATURES_STANDARD_BROWN_BEST, FEATURES_HORUS_TX_FULL)),
                           35: list(set().union(FEATURES_STANDARD_BROWN_BEST, FEATURES_HORUS_TX_FULL, FEATURES_HORUS_EMB_TX, FEATURES_HORUS_STATS_TX)),
                           36: list(set().union(FEATURES_STANDARD_BROWN_BEST, FEATURES_HORUS_CNN_CV + FEATURES_HORUS_CNN_TX)),
                           37: list(set().union(FEATURES_STANDARD_BROWN_BEST, FEATURES_HORUS_BASIC_AND_TX_CNN)),
                           38: list(set().union(FEATURES_STANDARD_BROWN_BEST, FEATURES_HORUS_CV_FULL)),
                           39: list(set().union(FEATURES_STANDARD_BROWN_BEST, FEATURES_HORUS_BASIC_AND_CV_CNN)),
                           40: list(set().union(FEATURES_STANDARD_BROWN_BEST, FEATURES_HORUS_TX_FULL, FEATURES_HORUS_CV_FULL)),
                           41: list(set().union(FEATURES_STANDARD_BROWN_BEST, FEATURES_HORUS_TX_FULL, FEATURES_HORUS_CV_FULL, FEATURES_HORUS_EMB_TX,
                                                FEATURES_HORUS_STATS_TX))
                           }
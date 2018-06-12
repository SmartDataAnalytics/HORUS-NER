import os

from src.config import HorusConfig

config = HorusConfig()
if config.root_dir == '':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    ROOT_DIR = config.root_dir

RUN_TAGGER_CMD = config.models_tweetnlp_java_param + " -jar " + config.models_tweetnlp_jar + " --model " + config.models_tweetnlp_model


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

PLO_KLASSES = {1: "LOC", 2: "ORG", 3: "PER"}
KLASSES = {1: "LOC", 2: "ORG", 3: "PER", 4: "O"}
KLASSES2 = {"LOC": 1, "ORG": 2, "PER": 3, "O": 4}

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


STANDARD_FEAT_BASIC_LEN = 60
STANDARD_FEAT_WORD_LEN = 15
STANDARD_FEAT_BROWN_LEN= 15
#----------------------------
STANDARD_FEAT_LEN = 90
'''
STANDARD FEATURES
'''

# 85:144 (60)
FEATURES_STANDARD = \
    range(HORUS_TOT_FEATURES + 1, (HORUS_TOT_FEATURES + 1 + STANDARD_FEAT_BASIC_LEN))

# 145:159 (15)
FEATURES_STANDARD_WORD = \
    range((HORUS_TOT_FEATURES + 1 + STANDARD_FEAT_BASIC_LEN),
          (HORUS_TOT_FEATURES + 1 + STANDARD_FEAT_BASIC_LEN + STANDARD_FEAT_WORD_LEN))

# 160:164 (5)
FEATURES_STANDARD_BROWN_64M_c320 = \
    range((HORUS_TOT_FEATURES + 1 + STANDARD_FEAT_BASIC_LEN + STANDARD_FEAT_WORD_LEN),
          (HORUS_TOT_FEATURES + 1 + STANDARD_FEAT_BASIC_LEN + STANDARD_FEAT_WORD_LEN) + 5)

# 165:169 (5)
FEATURES_STANDARD_BROWN_64M_c640 = \
    range((HORUS_TOT_FEATURES + 1 + STANDARD_FEAT_BASIC_LEN + STANDARD_FEAT_WORD_LEN) + 5,
          (HORUS_TOT_FEATURES + 1 + STANDARD_FEAT_BASIC_LEN + STANDARD_FEAT_WORD_LEN) + 5 + 5)

# 170:174 (5)
FEATURES_STANDARD_BROWN_500M_c1000 = \
    range((HORUS_TOT_FEATURES + 1 + STANDARD_FEAT_BASIC_LEN + STANDARD_FEAT_WORD_LEN) + 5 + 5,
          (HORUS_TOT_FEATURES + 1 + STANDARD_FEAT_BASIC_LEN + STANDARD_FEAT_WORD_LEN) + 5 + 5 + 5)

# = configuration 6
FEATURES_STANDARD_BROWN_BEST = FEATURES_STANDARD + FEATURES_STANDARD_WORD + FEATURES_STANDARD_BROWN_64M_c320
'''
HORUS FEATURES
'''
FEATURES_HORUS_BASIC_TX = [INDEX_TOT_RESULTS_TX, INDEX_TOT_TX_LOC, INDEX_TOT_TX_ORG, INDEX_TOT_TX_PER, INDEX_TOT_ERR_TRANS, INDEX_DIST_TX_I, INDEX_NR_RESULTS_SE_TX]
FEATURES_HORUS_CNN_TX = [INDEX_TOT_TX_LOC_TM_CNN, INDEX_TOT_TX_ORG_TM_CNN, INDEX_TOT_TX_PER_TM_CNN, INDEX_DIST_TX_I_TM_CNN, INDEX_TOT_TX_NONE_TM_CNN]

FEATURES_HORUS_BASIC_CV = [INDEX_TOT_IMG, INDEX_TOT_CV_LOC, INDEX_TOT_CV_ORG, INDEX_TOT_CV_PER, INDEX_DIST_CV_I, INDEX_PL_CV_I, INDEX_NR_RESULTS_SE_IMG]
FEATURES_HORUS_CNN_CV = [INDEX_TOT_CV_LOC_1_CNN, INDEX_TOT_CV_LOC_2_CNN, INDEX_TOT_CV_LOC_3_CNN, INDEX_TOT_CV_LOC_4_CNN, INDEX_TOT_CV_LOC_5_CNN, INDEX_TOT_CV_LOC_6_CNN, INDEX_TOT_CV_LOC_7_CNN, INDEX_TOT_CV_LOC_8_CNN, INDEX_TOT_CV_ORG_CNN, INDEX_TOT_CV_PER_CNN]

FEATURES_HORUS_EMB_TX = [INDEX_TOT_EMB_SIMILAR_LOC, INDEX_TOT_EMB_SIMILAR_ORG, INDEX_TOT_EMB_SIMILAR_PER, INDEX_TOT_EMB_SIMILAR_NONE]
FEATURES_HORUS_STATS_TX = [INDEX_TX_CNN_STAT_SUM_LOC, INDEX_TX_CNN_STAT_SUM_ORG, INDEX_TX_CNN_STAT_SUM_PER, INDEX_TX_CNN_STAT_SUM_NONE, INDEX_TX_CNN_STAT_AVG_LOC, INDEX_TX_CNN_STAT_AVG_ORG, INDEX_TX_CNN_STAT_AVG_PER, INDEX_TX_CNN_STAT_AVG_NONE, INDEX_TX_CNN_STAT_MAX_LOC, INDEX_TX_CNN_STAT_MAX_ORG, INDEX_TX_CNN_STAT_MAX_PER, INDEX_TX_CNN_STAT_MAX_NONE, INDEX_TX_CNN_STAT_MIN_LOC, INDEX_TX_CNN_STAT_MIN_ORG, INDEX_TX_CNN_STAT_MIN_PER, INDEX_TX_CNN_STAT_MIN_NONE, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_SUM_LOC, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_SUM_ORG, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_SUM_PER, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_SUM_NONE, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_AVG_LOC, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_AVG_ORG, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_AVG_PER, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_AVG_NONE, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MAX_LOC, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MAX_ORG, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MAX_PER, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MAX_NONE, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MIN_LOC, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MIN_ORG, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MIN_PER, INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MIN_NONE]

FEATURES_HORUS_TX = FEATURES_HORUS_BASIC_TX + FEATURES_HORUS_CNN_TX
FEATURES_HORUS_TX_EMB = FEATURES_HORUS_TX + FEATURES_HORUS_EMB_TX + FEATURES_HORUS_STATS_TX

FEATURES_HORUS_CV = FEATURES_HORUS_BASIC_CV + FEATURES_HORUS_CNN_CV

FEATURES_HORUS_BASIC_AND_CNN = FEATURES_HORUS_BASIC_TX + FEATURES_HORUS_CNN_TX + FEATURES_HORUS_BASIC_CV + FEATURES_HORUS_CNN_CV

FEATURES_HORUS = FEATURES_HORUS_TX + FEATURES_HORUS_CV

'''
BEST STANDARD FEATURES + HORUS FEATURES
'''
FEATURES_HORUS_BASIC_TX_BEST_STANDARD = FEATURES_HORUS_BASIC_TX + FEATURES_STANDARD_BROWN_BEST
FEATURES_HORUS_CNN_TX_BEST_STANDARD = FEATURES_HORUS_CNN_TX + FEATURES_STANDARD_BROWN_BEST
FEATURES_HORUS_BASIC_CV_BEST_STANDARD = FEATURES_HORUS_BASIC_CV + FEATURES_STANDARD_BROWN_BEST
FEATURES_HORUS_CNN_CV_BEST_STANDARD = FEATURES_HORUS_CNN_CV + FEATURES_STANDARD_BROWN_BEST
FEATURES_HORUS_EMB_TX_BEST_STANDARD = FEATURES_HORUS_EMB_TX + FEATURES_STANDARD_BROWN_BEST
FEATURES_HORUS_STATS_TX_BEST_STANDARD = FEATURES_HORUS_STATS_TX + FEATURES_STANDARD_BROWN_BEST
FEATURES_HORUS_TX_BEST_STANDARD = FEATURES_HORUS_TX + FEATURES_STANDARD_BROWN_BEST
FEATURES_HORUS_TX_EMB_BEST_STANDARD = FEATURES_HORUS_TX_EMB + FEATURES_STANDARD_BROWN_BEST
FEATURES_HORUS_CV_BEST_STANDARD = FEATURES_HORUS_CV + FEATURES_STANDARD_BROWN_BEST
FEATURES_HORUS_BASIC_AND_CNN_BEST_STANDARD = FEATURES_HORUS_BASIC_AND_CNN + FEATURES_STANDARD_BROWN_BEST
FEATURES_HORUS_BEST_STANDARD = FEATURES_HORUS + FEATURES_STANDARD_BROWN_BEST
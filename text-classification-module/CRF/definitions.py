import os

RUN_TAGGER_CMD = "java -XX:ParallelGCThreads=2 -Xmx2048m" + " -jar " + "/home/hady/PycharmProjects/text-classification-benchmarks/CRF/tweetnlp/ark-tweet-nlp-0.3.2.jar" + " --model " + "/home/hady/PycharmProjects/text-classification-benchmarks/CRF/tweetnlp/model.ritter_ptb_alldata_fixed.20130723"


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

NER_TAGS_ORG = ['ORG']
NER_TAGS_ORG.extend(NER_RITTER_ORG)
NER_TAGS_ORG.extend(NER_STANFORD_ORG)
NER_TAGS_ORG.extend(NER_NLTK_ORG)
NER_TAGS_ORG.extend(NER_CONLL_ORG)

NER_TAGS_LOC = ['LOC']
NER_TAGS_LOC.extend(NER_RITTER_LOC)
NER_TAGS_LOC.extend(NER_STANFORD_LOC)
NER_TAGS_LOC.extend(NER_NLTK_LOC)
NER_TAGS_LOC.extend(NER_CONLL_LOC)

#TODO: check if we have here ALL the NOUNs!!!
# merge of ALL noun tags, from all the POS taggers

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


seeds_dict_img_classes = {'per': ['person', 'human being', 'man', 'woman', 'human body', 'human face'],
                      'loc': ['location', 'place', 'volcano', 'stone', 'country', 'landscape', 'beach', 'sky', 'building', 'road', 'ocean', 'sea', 'lake', 'square', 'map', 'flag', 'city', 'forest', 'residence'],
                      'org': ['organisation', 'logo', 'logotype'],
                     'none': ['clipper', 'animal', 'telephone', 'time', 'cup', 'table', 'bottle', 'window', 'vehicle' 'monitor']}

seeds_dict_topics = {'per': ['arnett', 'david', 'richard', 'james', 'frank', 'george', 'misha',
                'student', 'education', 'coach', 'football', 'turkish',
                'albanian', 'romanian', 'professor', 'lawyer', 'president',
                'king', 'man', 'woman', 'danish', 'we', 'he', 'their', 'born',
                'directed', 'died', 'lives', 'boss', 'syrian', 'elected',
                'minister', 'candidate', 'daniel', 'robert', 'dude', 'guy',
                'girl', 'woman', 'husband', 'actor', 'people', 'celebrity', 'human'],
        'loc': ['china', 'usa', 'germany', 'leipzig', 'alaska', 'poland',
                'jakarta', 'kitchen', 'house', 'brazil', 'fuji', 'prison',
                'portugal', 'lisbon', 'france', 'oslo', 'airport', 'road',
                'highway', 'forest', 'sea', 'lake', 'stadium', 'hospital',
                'temple', 'beach', 'hotel', 'country', 'city', 'state', 'home',
                'world', 'mountain', 'landscape', 'island', 'land' ,'waterfall',
                'kitchen', 'room', 'office', 'bedroom', 'bathroom', 'hall', 'castle',
                'flag', 'map'],
        'org': ['microsoft', 'bloomberg', 'google', 'company', 'business', 'office',
                'contract', 'project', 'research', 'office', 'startup', 'organisation'
                'enterprise', 'venture', 'capital', 'milestones', 'risk',
                'funded', 'idea', 'industry', 'headquarters', 'product',
                'client', 'investment', 'certification', 'news', 'logo',
                'trademark', 'job', 'foundation'],
        'none': ['frog', 'animal', 'monkey', 'dog', 'skate', 'cup', 'money', 'cash',
                 'mouse', 'snake', 'telephone', 'glass', 'monitor', 'bible', 'book',
                 'dictionary', 'religion', 'politics', 'sports', 'question', 'linux',
                 'java', 'python', 'months', 'time', 'wallet', 'umbrella', 'cable',
                 'internet', 'connection', 'pencil', 'earphone', 'shopping', 'buy',
                 'headphones', 'bread', 'food', 'cake', 'bottle', 'table', 'jacket',
                 'politics', 'computer', 'laptop', 'blue', 'green', 'bucket', 'orange', 'rose',
                 'key', 'clock', 'connector']}

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
INDEX_TOT_IMG = 11
INDEX_TOT_CV_LOC = 12
INDEX_TOT_CV_ORG = 13
INDEX_TOT_CV_PER = 14
INDEX_DIST_CV_I = 15
INDEX_PL_CV_I= 16
INDEX_NR_RESULTS_SE_IMG = 17
INDEX_MAX_KLASS_PREDICT_CV = 18
INDEX_TOT_RESULTS_TX = 19
INDEX_TOT_TX_LOC = 20
INDEX_TOT_TX_ORG = 21
INDEX_TOT_TX_PER = 22
INDEX_TOT_ERR_TRANS = 23
INDEX_DIST_TX_I = 24
INDEX_NR_RESULTS_SE_TX = 25
INDEX_MAX_KLASS_PREDICT_TX = 26
INDEX_INDEX_START_TERM = 27
INDEX_TOT_TX_LOC_TM_CNN = 28
INDEX_TOT_TX_ORG_TM_CNN = 29
INDEX_TOT_TX_PER_TM_CNN = 30
INDEX_DIST_TX_I_TM_CNN = 31
INDEX_TOT_CV_LOC_1_CNN = 32
INDEX_TOT_CV_ORG_CNN = 33
INDEX_TOT_CV_PER_CNN = 34
INDEX_TOT_CV_LOC_2_CNN = 35
INDEX_TOT_CV_LOC_3_CNN = 36
INDEX_TOT_CV_LOC_4_CNN = 37
INDEX_MAX_KLASS_PREDICT_TX_CNN = 38
INDEX_MAX_KLASS_PREDICT_COMPOUND = 39
INDEX_KLASS_FINAL_MODEL = 40
INDEX_MAX_KLASS_PREDICT_CV_CNN = 41
INDEX_TOT_EMB_SIMILAR_LOC = 42
INDEX_TOT_EMB_SIMILAR_ORG = 43
INDEX_TOT_EMB_SIMILAR_PER = 44
INDEX_TOT_CV_LOC_5_CNN = 45
INDEX_TOT_CV_LOC_6_CNN = 46
INDEX_TOT_CV_LOC_7_CNN = 47
INDEX_TOT_CV_LOC_8_CNN = 48
INDEX_TOT_EMB_SIMILAR_NONE = 49
INDEX_TOT_TX_NONE_TM_CNN = 50
INDEX_TARGET_NER = 51

HORUS_TOT_FEATURES = 52

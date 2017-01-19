import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_PATH = os.path.join(ROOT_DIR, 'src/horus/experiments')
HORUS_PATH = os.path.join(ROOT_DIR, 'src/horus/components')
OUTPUT_PATH = os.path.join(ROOT_DIR, 'output')
RESOURCES_PATH = os.path.join(ROOT_DIR, 'src/horus/resource')
DATASET_PATH = os.path.join(ROOT_DIR, 'data/dataset')
POS_TAGGER_PATH = os.path.join(ROOT_DIR, 'src/horus/postagger')

NER_RITTER_PER = ['B-person', 'I-person']
NER_RITTER_ORG = ['B-company', 'I-company']
NER_RITTER_LOC = ['B-geo-loc', 'I-geo-loc']

NER_RITTER = []
NER_RITTER.extend(NER_RITTER_PER)
NER_RITTER.extend(NER_RITTER_ORG)
NER_RITTER.extend(NER_RITTER_LOC)

KLASSES = {1: "LOC", 2: "ORG", 3: "PER"}

CMU_PENN_TAGS = [['N', 'NNS'], ['O', 'PRP'], ['S', 'PRP$'], ['^', 'NNP'], ["D", "DT"], ["A", "JJ"], ["P", "IN"],
                     ["&", "CC"],["V", "VBD"], ["R", "RB"], ["!", "UH"], ["T", "RP"], ["$", "CD"]]

CMU_UNI_TAGS = [["N", "NOUN"], ["^", "NOUN"], ["V", "VERB"], ["D", "DET"], ["A", "ADJ"], ["P", "ADP"],
                        ["&", "CCONJ"], ["R", "ADV"], ["!", "INTJ"], ["O","PRON"], ["$", "NUM"], [",", "PUNCT"]]
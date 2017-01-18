import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_PATH = os.path.join(ROOT_DIR, 'src/horus/experiments')
HORUS_PATH = os.path.join(ROOT_DIR, 'src/horus/components')
OUTPUT_PATH = os.path.join(ROOT_DIR, 'output')
RESOURCES_PATH = os.path.join(ROOT_DIR, 'src/horus/resource')
DATASET_PATH = os.path.join(ROOT_DIR, 'data/dataset')

NER_RITTER_PER = ['B-person', 'I-person']
NER_RITTER_ORG = ['B-company', 'I-company']
NER_RITTER_LOC = ['B-geo-loc', 'I-geo-loc']

NER_RITTER = []
NER_RITTER.extend(NER_RITTER_PER)
NER_RITTER.extend(NER_RITTER_ORG)
NER_RITTER.extend(NER_RITTER_LOC)

KLASSES = {1: "LOC", 2: "ORG", 3: "PER"}

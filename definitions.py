import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_PATH = os.path.join(ROOT_DIR, 'src/experiments')
HORUS_PATH = os.path.join(ROOT_DIR, 'src/components')
OUTPUT_PATH = os.path.join(ROOT_DIR, 'output')
RESOURCES_PATH = os.path.join(ROOT_DIR, 'src/resource')
DATASET_PATH = os.path.join(ROOT_DIR, 'src/dataset')

NER_RITTER_PER = ['B-person', 'I-person']
NER_RITTER_ORG = ['B-company', 'I-company']
NER_RITTER_LOC = ['B-geo-loc', 'I-geo-loc']

NER_RITTER = []
NER_RITTER.extend(NER_RITTER_PER)
NER_RITTER.extend(NER_RITTER_ORG)
NER_RITTER.extend(NER_RITTER_LOC)

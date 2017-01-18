import os
from horus.components.config import HorusConfig

config = HorusConfig()
if config.root_dir == '':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    ROOT_DIR = config.root_dir

EXPERIMENTS_PATH = os.path.join(ROOT_DIR, 'experiments')
HORUS_PATH = os.path.join(ROOT_DIR, 'components')
RESOURCES_PATH = os.path.join(ROOT_DIR, 'resource')


NER_RITTER_PER = ['B-person', 'I-person']
NER_RITTER_ORG = ['B-company', 'I-company']
NER_RITTER_LOC = ['B-geo-loc', 'I-geo-loc']

NER_RITTER = []
NER_RITTER.extend(NER_RITTER_PER)
NER_RITTER.extend(NER_RITTER_ORG)
NER_RITTER.extend(NER_RITTER_LOC)

KLASSES = {1: "LOC", 2: "ORG", 3: "PER"}

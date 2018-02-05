from horus.core.config import HorusConfig
from horus.core.feature_extraction.main import FeatureExtraction

extractor = FeatureExtraction()
print extractor.config.version_label

exp_folder = 'experiments/EXP_002/'

ret = extractor.extract_features('Ritter/ner.txt', exp_folder, 'ritter')
ret = extractor.extract_features('wnut/2016.conll.freebase.ascii.txt', exp_folder, 'wnut15')
ret = extractor.extract_features('wnut/2015.conll.freebase', exp_folder, 'wnut16')
# attention: change POS tag lib in the HORUS.ini to NLTK before run this
ret = extractor.extract_features('coNLL2003/nodocstart_coNLL2003.eng.testA', exp_folder, 'conll03', 0, 3)

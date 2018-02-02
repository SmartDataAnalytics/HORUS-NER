from horus.core.config import HorusConfig
from horus.core.service import Core

horus = Core()
print horus.version_label
config = HorusConfig()

exp_folder = 'experiments/EXP_002/'

#ret = horus.annotate(None, config.dataset_path + "Ritter/ner.txt", 1, exp_folder + "ritter.horus", "csv", 'ritter')
ret = horus.annotate(None, config.dataset_path + "wnut/2016.conll.freebase", 1, exp_folder + "out_exp003_wnut16_en_tweetNLP", "csv", 'wnut2016')
#ret = horus.annotate(None, config.dataset_path + "wnut/2015.conll.freebase", 1, exp_folder + "out_exp003_wnut15_en_tweetNLP", "csv", 'wnut2015')
#ret = horus.annotate(None, config.dataset_path + "coNLL2003/nodocstart_coNLL2003.eng.testA", 1, exp_folder + "out_exp003_coNLL2003testA_en_NLTK", "csv", 'coNLL2003testA', 0, 3)

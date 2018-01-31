from horus.core.config import HorusConfig
from horus.core.service import Core

horus = Core()
print horus.version_label
config = HorusConfig()


ret = horus.annotate(None, config.dataset_path + "Ritter/ner_sample_ritter.txt", 1, "experiments/EXP_002/ritter.horus", "csv", 'ritter')
#ret = horus.annotate(None, config.dataset_path + "wnut/2016.conll.freebase.ascii.txt", 1, "experiments/EXP_do_tokenization/out_exp003_wnut16_en_tweetNLP", "csv", 'wnut2016')
#ret = horus.annotate(None, config.dataset_path + "wnut/2015.conll.freebase", 1, "experiments/EXP_do_tokenization/out_exp003_wnut15_en_tweetNLP", "csv", 'wnut2015')
#ret = horus.annotate(None, config.dataset_path + "coNLL2003/nodocstart_coNLL2003.eng.testA", 1, "experiments/EXP_001/out_exp003_coNLL2003testA_en_NLTK", "csv", 'coNLL2003testA', 0, 3)
print "done!"
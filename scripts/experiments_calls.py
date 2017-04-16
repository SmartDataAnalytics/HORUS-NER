from horus.components.config import HorusConfig
from horus.components.core import Core

horus = Core(False, 5)
print horus.version_label
config = HorusConfig()
#ret = horus.annotate(None, config.dataset_path + "Ritter/ner_one_sentence.txt", 1, "experiments/ritter/EXP_001/out_exp001_22_tweetNLP", "csv", 'ritter')
#ret = horus.annotate(None, config.dataset_path + "Ritter/ner.txt", 1, "experiments/ritter/EXP_001/out_exp001_3_tweetNLP", "csv", 'ritter')
#ret = horus.annotate(None, config.dataset_path + "wnut/2015.conll.freebase", 1, config.output_path + "experiments/ritter/EXP_001/out_exp002_wnut_tweetNLP", "csv", 'wnut2015')
ret = horus.annotate(None, config.dataset_path + "wnut/2016.conll.freebase", 1, "experiments/ritter/EXP_001/out_exp003_wnut16_tweetNLP", "csv", 'wnut2016')
print "done!"
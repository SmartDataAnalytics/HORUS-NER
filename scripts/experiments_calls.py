from horus.components.config import HorusConfig
from horus.components.core import Core

horus = Core(False, 5)
print horus.version_label
config = HorusConfig()
ret = horus.annotate(None, config.dataset_path + "Ritter/ner_sample_ritter.txt", 1, "experiments/ritter/EXP_001/out_exp001_1", "csv", 'ritter')
#ret = horus.annotate(None, config.dataset_path + "wnut/2015.conll.freebase", 1, config.output_path + "experiments/ritter/EXP_000/out_exp000_5", "csv", 'wnut2015')
#ret = horus.annotate(None, config.dataset_path + "wnut/2016.conll.freebase", 1, config.output_path + "experiments/ritter/EXP_000/out_exp000_6", "csv", 'wnut2016')
print "done!"
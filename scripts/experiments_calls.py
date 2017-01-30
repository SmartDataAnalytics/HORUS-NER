from horus.components.config import HorusConfig
from horus.components.core import Core

horus = Core(False, 5)
print horus.version_label
config = HorusConfig()
ret = horus.annotate(None, config.dataset_path + "Ritter/ner.txt", 1,
                     config.output_path + "experiments/ritter/EXP_000/out_exp000_5", "csv")
print "done!"





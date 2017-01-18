from horusner.components.config import HorusConfig
from horusner.components.core import Core

horus = Core(False, 5)
print horus.version_label
config = HorusConfig()
ret = horus.annotate(None, config.dataset_path + "Ritter/ritter_ner.tsv", 1,
                     config.output_path + "experiments/ritter/EXP_000/out_exp000_1", "csv")


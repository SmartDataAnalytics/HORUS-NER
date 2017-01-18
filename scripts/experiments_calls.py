from horusner.components.core import Core

import definitions

horus = Core(False, 5)
print horus.version_label
ret = horus.annotate(None, definitions.DATASET_PATH + "/Ritter/ritter_ner.tsv", 1,
                     definitions.OUTPUT_PATH + "/experiments/ritter/EXP_000/out_exp000_1", "csv")


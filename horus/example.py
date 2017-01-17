from horus.core import Core

horus = Core(False, 5)
print horus.version_label
print horus.annotate("coca cola has a strange flavor", "", 0, "../output/out", "csv")
print horus.get_cv_annotation()

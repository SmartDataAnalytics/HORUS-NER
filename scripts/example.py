from horus.components.core import Core

horus = Core(False, 5)
print horus.version_label
print horus.annotate("going to nyc today and san francisco afterwards", "", 0, "../output/out3", "csv")
print horus.get_cv_annotation()

from horus.components.core import Core

horus = Core(False, 5)
print horus.version_label
print horus.annotate("petropolis is more relaxing than bangu", "", 0, "../output/out", "csv")
print horus.get_cv_annotation()

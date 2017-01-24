from horus.components.core import Core

horus = Core(False, 5)
print horus.version_label
print horus.annotate("vasco da gama is a football team from rio de janeiro", "", 0, "../output/out1", "csv")
print horus.get_cv_annotation()

from langdetect import detect
from langdetect import detect_langs
from langdetect import DetectorFactory

# https://github.com/Mimino666/langdetect
# to ensure deterministic behaviour
DetectorFactory.seed = 0

print detect("War doesn't show who's right, just who's left.")
print detect("Ein, zwei, drei, vier")
print detect_langs("Otec matka syn.")
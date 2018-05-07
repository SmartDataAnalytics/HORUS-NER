import nltk
try:
    nltk.data.find('averaged_perceptron_tagger.zip')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('punkt.zip')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('maxent_ne_chunker.zip')
except LookupError:
    nltk.download('maxent_ne_chunker')
try:
    nltk.data.find('universal_tagset.zip')
except LookupError:
    nltk.download('universal_tagset')
try:
    nltk.data.find('words.zip')
except LookupError:
    nltk.download('words')
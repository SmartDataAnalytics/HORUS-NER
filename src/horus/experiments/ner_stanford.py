from nltk import word_tokenize
from nltk.tag.stanford import StanfordNERTagger

st = StanfordNERTagger('/usr/share/stanford-ner/classifiers/all.3class.distsim.crf.ser.gz',
               '/usr/share/stanford-ner/stanford-ner.jar')
text = 'diego esteves'
tag1 = st.tag(text.split())
tag2 = st.tag(word_tokenize(text))
print tag1, tag2



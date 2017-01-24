from nltk import word_tokenize
from nltk.tag.stanford import StanfordNERTagger

st = StanfordNERTagger('/Users/esteves/Github/horus-models/src/horus/resource/models/stanford/english.all.3class.distsim.crf.ser.gz',
               '/Users/esteves/Github/horus-models/src/horus/resource/models/stanford/stanford-ner.jar')
text = 'diego esteves is a nice guy'
#tag1 = st.tag(text.split())
tag2 = st.tag(word_tokenize(text))
print #tag1
print tag2



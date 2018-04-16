import spacy
import en_core_web_sm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


nlp = en_core_web_sm.load()
#spacy.load('en')

import shorttext
emb = '/Volumes/dne5data/embeddings/GoogleNews-vectors-negative300.bin.gz'
#emb = '/Users/diegoesteves/Downloads/GoogleNews-vectors-negative300 (1).bin'

dict = {'per': ['arnett', 'david', 'richard', 'james', 'frank', 'george', 'misha',
                'students', 'education', 'coach', 'football', 'turkish',
                'albanian', 'romanian', 'professor', 'lawyer', 'president',
                'king', 'man', 'woman', 'danish', 'we', 'he', 'their', 'born',
                'directed', 'died', 'lives', 'boss', 'syrian', 'elected',
                'minister', 'candidate', 'daniel', 'robert', 'dude', 'guy',
                'girl', 'woman', 'husband', 'actor', 'people', 'celebrity'],
        'loc': ['china', 'usa', 'germany', 'leipzig', 'alaska', 'poland',
                'jakarta', 'kitchen', 'house', 'brazil', 'fuji', 'prison',
                'portugal', 'lisbon', 'france', 'oslo', 'airport', 'road',
                'highway', 'forest', 'sea', 'lake', 'stadium', 'hospital',
                'temple', 'beach', 'hotel', 'country', 'city', 'state',
                'world', 'mountain', 'landscape', 'island', 'land'],
        'org': ['microsoft', 'bloomberg', 'google', 'company', 'business',
                'contract', 'project', 'research', 'office', 'startup',
                'enterprise', 'venture', 'capital', 'milestones', 'risk',
                'funded', 'idea', 'industry', 'headquarters', 'product',
                'client', 'investment', 'users', 'certification', 'news']}

wvmodel = shorttext.utils.load_word2vec_model(emb)
#trainclassdict = shorttext.data.subjectkeywords()
trainclassdict  = dict



classifier1 = shorttext.classifiers.load_varnnlibvec_classifier(wvmodel, 'text_convnet_plo.bin')
classifier2 = shorttext.classifiers.load_varnnlibvec_classifier(wvmodel, 'text_clstm_word_embed.bin')
classifier3 = shorttext.classifiers.load_varnnlibvec_classifier(wvmodel, 'text_double_cnn_word_embed.bin')


kmodel1 = shorttext.classifiers.frameworks.CNNWordEmbed(len(trainclassdict.keys()), vecsize=wvmodel.vector_size)
kmodel2 = shorttext.classifiers.frameworks.CLSTMWordEmbed(len(trainclassdict.keys()), vecsize=wvmodel.vector_size)
kmodel3 = shorttext.classifiers.frameworks.DoubleCNNWordEmbed(len(trainclassdict.keys()), vecsize=wvmodel.vector_size)

classifier1 = shorttext.classifiers.VarNNEmbeddedVecClassifier(wvmodel)
classifier2 = shorttext.classifiers.VarNNEmbeddedVecClassifier(wvmodel)
classifier3 = shorttext.classifiers.VarNNEmbeddedVecClassifier(wvmodel)

print trainclassdict.keys()
try:
    classifier1.train(trainclassdict, kmodel1, nb_epoch=1000)
    classifier2.train(trainclassdict, kmodel1, nb_epoch=1000)
    classifier3.train(trainclassdict, kmodel1, nb_epoch=1000)
except:
    raise


classifier1.save_compact_model('text_cnn_word_embed_plo.bin')
classifier2.save_compact_model('text_clstm_word_embed.bin')
classifier3.save_compact_model('text_double_cnn_word_embed.bin')

print classifier1.score('paris hilton was once the toast of the townMaradona Franco is an Argentine retired professional footballer. He has served as a manager and coach at other clubs as well as the national team of Argentina. Many in the sport, including football writers, players, and fans, regard Maradona as the greatest football player of all time.')
print classifier1.score('river')
print classifier1.score('chile')
print classifier1.score('jack')
print classifier1.score('esteves')
print classifier1.score('global co.')
print classifier1.score('global solutions')
exit(0)
import spacy
import en_core_web_sm
import os
import shorttext
from src.config import HorusConfig
import tensorflow as tf

from src.util.nlp_tools import NLPTools

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
nlp = en_core_web_sm.load()
#spacy.load('en')

class TopicModelingShortCNN():
    def __init__(self, config, w2v, mode='test'):
        try:
            self.config = config
            self.config.logger.info('loading TopicModelingShortCNN')
            self.wvmodel = w2v
            self.mode = mode
            self.trainclassdict = {'per': ['arnett', 'david', 'richard', 'james', 'frank', 'george', 'misha',
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
                'temple', 'beach', 'hotel', 'country', 'city', 'state', 'home',
                'world', 'mountain', 'landscape', 'island', 'land' ,'waterfall',
                'kitchen', 'room', 'office', 'bedroom', 'bathroom', 'hall', 'castle',
                'flag', 'map'],
        'org': ['microsoft', 'bloomberg', 'google', 'company', 'business',
                'contract', 'project', 'research', 'office', 'startup',
                'enterprise', 'venture', 'capital', 'milestones', 'risk',
                'funded', 'idea', 'industry', 'headquarters', 'product',
                'client', 'investment', 'certification', 'news', 'logo',
                'trademark', 'job', 'foundation'],
        'none': ['frog', 'animal', 'monkey', 'dog', 'skate', 'cup', 'money', 'cash',
                 'mouse', 'snake', 'telephone', 'glass', 'monitor', 'bible', 'book',
                 'dictionary', 'religion', 'politics', 'sports', 'question', 'linux',
                 'java', 'python', 'months', 'time', 'wallet', 'umbrella', 'cable',
                 'internet', 'connection', 'pencil', 'earphone', 'shopping', 'buy',
                 'headphones', 'bread', 'food', 'cake', 'bottle', 'table', 'jacket',
                 'politics', 'computer', 'laptop', 'blue', 'green', 'bucket', 'orange', 'rose',
                 'key', 'clock', 'connector']}
            #self.config.logger.debug('loading embeedings')
            #self.wvmodel = shorttext.utils.load_word2vec_model(config.embeddings_path)
            if mode=='test':
                self.classifier = shorttext.classifiers.load_varnnlibvec_classifier(self.wvmodel, config.models_1_text_cnn)
            self.graph = tf.get_default_graph()
        except Exception as e:
            raise e


    def predict(self, text):
        try:
            with self.graph.as_default():
                return self.classifier.score(text)
        except:
            raise

    def detect_text_klass(self, text):
        predictions = []
        try:
            dict = self.classifier.score(text)
            predictions.append(dict.get('loc'))
            predictions.append(dict.get('org'))
            predictions.append(dict.get('per'))
            predictions.append(dict.get('none'))
            predictions.append(0)
            return predictions
        except Exception as e:
            raise e

    def train(self, epochs=5000):
        try:
            if self.mode == 'test':
                raise Exception('error, set param "mode=train"')

            print(self.trainclassdict.keys())

            kmodel1 = shorttext.classifiers.frameworks.CNNWordEmbed(len(self.trainclassdict.keys()), vecsize=self.wvmodel.vector_size)
            kmodel2 = shorttext.classifiers.frameworks.CLSTMWordEmbed(len(self.trainclassdict.keys()), vecsize=self.wvmodel.vector_size)
            kmodel3 = shorttext.classifiers.frameworks.DoubleCNNWordEmbed(len(self.trainclassdict.keys()), vecsize=self.wvmodel.vector_size)

            classifier1 = shorttext.classifiers.VarNNEmbeddedVecClassifier(self.wvmodel)
            classifier2 = shorttext.classifiers.VarNNEmbeddedVecClassifier(self.wvmodel)
            classifier3 = shorttext.classifiers.VarNNEmbeddedVecClassifier(self.wvmodel)

            classifier1.train(self.trainclassdict, kmodel1, nb_epoch=epochs)
            classifier2.train(self.trainclassdict, kmodel2, nb_epoch=epochs)
            classifier3.train(self.trainclassdict, kmodel3, nb_epoch=epochs)

            classifier1.save_compact_model(self.config.dir_models + 'cnn_topic_modeling/4classes_text_cnn_word_embed_plo.bin')
            classifier2.save_compact_model(self.config.dir_models + 'cnn_topic_modeling/4classes_text_clstm_word_embed.bin')
            classifier3.save_compact_model(self.config.dir_models + 'cnn_topic_modeling/4classes_text_double_cnn_word_embed.bin')

        except:
            raise


if __name__ == '__main__':

    config = HorusConfig()
    tools = NLPTools(config)
    topic = TopicModelingShortCNN(config, tools.word2vec_google, mode='test')
    #topic.train()


    print(topic.predict('orlando'))
    print(topic.predict('river'))
    print(topic.predict('chile'))
    print(topic.predict('jack'))
    print(topic.predict('esteves'))
    print(topic.predict('global co.'))
    print(topic.predict('global solutions'))
    print(topic.predict('jack watson'))

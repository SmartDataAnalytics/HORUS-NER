from src.classifiers.computer_vision.inception import InceptionCV
from src.classifiers.computer_vision.places365 import Places365CV
from src.classifiers.text_classification.topic_modeling_short_cnn import TopicModelingShortCNN
from src.classifiers.util.inception import imagenet
from src.config import HorusConfig

import re

from src.core.util import definitions
from src.core.util.nlp_tools import NLPTools

config = HorusConfig()

print('loading places 365 CNN model...')
placesCNN = Places365CV(config)

print('loading inception model...')
incep_model = InceptionCV(config)

print('getting imageNET labels...')
class_names = imagenet.create_readable_names_for_imagenet_labels()
print('done')


pos_img_places = ['18_0_3', '18_0_4', '18_0_5', '18_0_9', '18_0_10', '20_0_1', '20_0_2', '20_0_3', '20_0_4', '20_0_5',
                  '20_0_6', '20_0_7', '20_0_8', '20_0_9', '20_0_10', '38_0_1', '38_0_2', '38_0_3']
pos_img_logos = ['28_0_1', '28_0_2', '28_0_3', '28_0_10', '68_0_8', '102_0_10', '104_0_8', '104_0_4', '114_0_8', '134_0_4',
                 '188_0_1', '188_0_2', '188_0_4', '188_0_5', '198_0_2', '200_0_4']

pos_img_per = ['4_0_3', '4_0_4', '4_0_5', '4_0_6', '4_0_7', '6_0_7', '6_0_6', '8_0_6', '6_0_8']

#model.predict('4_0_1.jpg')
#model.predict('10_0_4.jpg')
#model.predict('10_0_5.jpg')
#model.predict('10_0_7.jpg')
#model.predict('18_0_2.jpg')
#model.predict('16_0_9.jpg')
_version = 'V4'
print('---------------------')

extended_seeds_PER = []
from nltk.corpus import wordnet as wn


for k, V in definitions.seeds_dict_img_classes.iteritems():
    for v in V:
        for i, syns in enumerate(wn.synsets(v)):
            # print(' hypernyms = ', ' '.join(list(chain(*[l.lemma_names() for l in syns.hypernyms()]))))
            # print(' hyponyms = ', ' '.join(list(chain(*[l.lemma_names() for l in syns.hyponyms()]))))
            _s = wn.synset_from_pos_and_offset(syns.pos(), syns.offset())

            extended_seeds_PER.append(str(_s._name).split('.')[0])
            extended_seeds_PER.extend([str(_s._name).split('.')[0] for _s in syns.hypernyms()])

print(extended_seeds_PER)
exit(0)
print('loading utils...')
utils = NLPTools(config)
print('loading topic model CNN model...')
topicCNN = TopicModelingShortCNN(w2v=utils.word2vec_google, config=config, mode='test')

for img in pos_img_per:
    try:
        try:
            print('predicting places365...')
            prediction_values_places365 = placesCNN.predict(img + '.jpg')
        except Exception as e:
            print('error..')
            pass

        try:
            print('predicting inception...')
            prediction_values_inception = incep_model.predict(img + '.jpg', top=5)
        except Exception as e:
            print('error..')
            pass

        K = []
        tot_loc = 0
        tot_per = 0
        tot_org = 0
        tot_none = 0
        i = 0
        # image classification CNN
        #print('places365', prediction_values_places365)
        #print('inception', prediction_values_inception)
        #print('')

        final_label_prob_vector = prediction_values_inception
        final_label_prob_vector.extend(prediction_values_places365)
        print('final', final_label_prob_vector)



    except Exception as e:
        print(e)

print('OK')
    #    probability = prediction_values[i][1]
    #    print("{}: {:.2f}%".format(predicted_class, probability * 100))


    #model.predict('12_0_2.jpg', _version)
    #model.predict('12_0_3.jpg', _version)
    #model.predict('12_0_6.jpg', _version)
    #model.predict('14_0_6.jpg', _version)
    #model.predict('22_0_3.jpg', _version)
    #model.predict('22_0_5.jpg', _version)
    #print('---------------------')
    #model.predict('38_0_2.jpg', _version)
    #model.predict('50_0_3.jpg', _version)
    #model.predict('50_0_4.jpg', _version)
    #model.predict('62_0_10.jpg', _version)
    #model.predict('62_0_6.jpg', _version)
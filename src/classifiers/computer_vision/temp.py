from src.classifiers.computer_vision.inception import InceptionCV
from src.classifiers.text_classification.topic_modeling_cnn import TopicModelingShortCNN
from src.classifiers.util.inception import imagenet
from src.config import HorusConfig
from nltk.corpus import wordnet as wn

config = HorusConfig()

print('loading topic model CNN model...')
topicCNN = TopicModelingShortCNN(config)
print('loading inception model...')
incep_model = InceptionCV(config)
print('getting imageNET labels...')
class_names = imagenet.create_readable_names_for_imagenet_labels()
print('done')

ids_PER = []
ids_ORG = []
ids_LOC = []

for k, V in incep_model.seeds.iteritems():
    for v in V:
        for i, syns in enumerate(wn.synsets(v)):
            #print(' hypernyms = ', ' '.join(list(chain(*[l.lemma_names() for l in syns.hypernyms()]))))
            #print(' hyponyms = ', ' '.join(list(chain(*[l.lemma_names() for l in syns.hyponyms()]))))
            _s = wn.synset_from_pos_and_offset(syns.pos(), syns.offset())
            if k == 'PER':
                ids_PER.append(_s)
                #ids_PER.extend(syns.hypernyms())
                #ids_PER.extend(syns.hyponyms())
            elif k == 'ORG':
                ids_ORG.append(_s)
                #ids_ORG.extend(syns.hypernyms())
                #ids_ORG.extend(syns.hyponyms())
            elif k == 'LOC':
                ids_LOC.append(_s)
                #ids_LOC.extend(syns.hypernyms())
                #ids_LOC.extend(syns.hyponyms())
            else:
                raise('key error')


print(set(ids_PER))
print(set(ids_LOC))
print(set(ids_ORG))

pos_img_places = ['18_0_3', '18_0_4', '18_0_5', '18_0_9', '18_0_10', '20_0_1', '20_0_2', '20_0_3', '20_0_4', '20_0_5',
                  '20_0_6', '20_0_7', '20_0_8', '20_0_9', '20_0_10', '38_0_1', '38_0_2', '38_0_3', '38_0_4']
dict_img_places = []

#model.predict('4_0_1.jpg')
#model.predict('10_0_4.jpg')
#model.predict('10_0_5.jpg')
#model.predict('10_0_7.jpg')
#model.predict('18_0_2.jpg')
#model.predict('16_0_9.jpg')
_version = 'V4'
print('---------------------')

for img in pos_img_places:
    prediction_values = incep_model.predict(img + '.jpg', top=5)
    K = []
    tot_loc = 0
    tot_per = 0
    tot_org = 0
    tot_none = 0
    i = 0
    for i in range(0,len(prediction_values)):
        labels = class_names[prediction_values[i][0]]
        probability = prediction_values[i][1]
        for l in labels.split(','):
            i+=1
            d = topicCNN.predict(l)
            tot_loc += (d.get('loc') * probability)
            tot_per += (d.get('per') * probability)
            tot_org += (d.get('org') * probability)
            tot_none += (d.get('none') * probability)

    print('img', img, 'LOC', tot_loc/i, 'ORG', tot_org/i, 'PER', tot_per/i, 'NONE', tot_none/i)

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
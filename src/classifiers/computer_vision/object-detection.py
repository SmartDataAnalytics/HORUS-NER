import os
import cv2
import numpy as np
from os.path import join
from sklearn import svm
from sklearn.externals import joblib


root = os.getcwd()
root_img = root + '/img/'
ner_class = 'loc'
feature = 'map2'

CONST_DATA_POS_PATH      = root_img + ner_class + '/train/pos/' + feature
CONST_DATA_NEG_PATH      = root_img + ner_class + '/train/neg'
CONST_TEST_DATA_POS_PATH = root_img + ner_class + '/test/pos/' + feature + '/'
CONST_TEST_DATA_NEG_PATH = root_img + ner_class + '/test/neg/'

CONST_SVM_MODEL_PATH = root + '/models/' + ner_class + '_' + feature + '_svm2.pkl'
CONST_PREDICTIONS = root + '/predictions/EXP001/' + ner_class + '_' + feature + '_svm_001.txt'
CONST_DICTIONARY_PATH = root + '/models/dict_002_' + ner_class + '_' + feature + '.pkl'
CONST_CREATE_DICTIONARY = 0
CONST_RETRAIN_MODEL = 1


#Store and Retrieve keypoint features
#    temp_array = []
#    temp = pickle_keypoints(kp1, desc1)
#    temp_array.append(temp)
#    temp = pickle_keypoints(kp2, desc2)
#    temp_array.append(temp)
#    pickle.dump(temp_array, open("keypoints_database.p", "wb"))
#
#    #Retrieve Keypoint Features
#    keypoints_database = pickle.load( open( "keypoints_database.p", "rb" ) )
#    kp1, desc1 = unpickle_keypoints(keypoints_database[0])
#    kp1, desc1 = unpickle_keypoints(keypoints_database[1])

def pickle_keypoints(keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id, descriptors[i])
        ++i
        temp_array.append(temp)
    return temp_array

def unpickle_keypoints(array):
    keypoints = []
    descriptors = []
    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors)


def temp_create_image_keypoints(fn):
    img = cv2.imread(fn)
    gray = cv2.imread(fn, 1)
    sift = cv2.xfeatures2d.StarDetector_create()
    dummy = np.zeros((1, 1))
    kp = sift.detect(gray, None)
    #img = cv2.drawKeypoints(gray, kp, dummy, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img = cv2.drawKeypoints(gray, kp, dummy, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('star_keypoints.jpg', img)



temp_create_image_keypoints('/Users/esteves/Github/horus-models/src/horus/resource/osi_black_and_white_light_backgrounds_logo.png')
exit(0)

def path(cls,i):
    if cls == "pos-":
        return "%s/%s%d.jpg" % (CONST_DATA_POS_PATH, cls, i+1)
    else:
        return "%s/%s%d.jpg" % (CONST_DATA_NEG_PATH, cls, i+1)

def extract_sift(fn):
    print 'sift: ' + fn
    im = cv2.imread(fn, 0)
    return extract.compute(im, detect.detect(im))[1]

def bow_features(fn):
    print 'bow: ' + fn
    im = cv2.imread(fn, 0)
    return extract_bow.compute(im, detect.detect(im))

def predict(fn):
    f = bow_features(fn); p = svm.predict(f)
    #print fn, "\t", p[1][0][0]
    print fn, "\t", p[0]
    return p

print '- starting the process'
pos, neg = "pos-", "neg-"

detect = cv2.xfeatures2d.SIFT_create()
extract = cv2.xfeatures2d.SIFT_create()

flann_params = dict(algorithm=1, trees=5)
flann = cv2.FlannBasedMatcher(flann_params, {})

bow_kmeans_trainer = cv2.BOWKMeansTrainer(40)
extract_bow = cv2.BOWImgDescriptorExtractor(extract, flann)

if CONST_CREATE_DICTIONARY == 1:
    print '- adding features to cluster'
    for i in range(100): #8
        bow_kmeans_trainer.add(extract_sift(path(pos, i)))
        bow_kmeans_trainer.add(extract_sift(path(neg, i)))

    print '- clustering'
    voc = bow_kmeans_trainer.cluster()
    joblib.dump(voc, CONST_DICTIONARY_PATH, compress=3)
else:
    print '- loading the vocabulary'
    voc = joblib.load(CONST_DICTIONARY_PATH)

print '- setting vocabulary to extractor'
extract_bow.setVocabulary(voc)

if CONST_RETRAIN_MODEL == 1:
    print '- creating feature_extraction data'
    traindata, trainlabels = [], []
    for i in range(213): # 20
        traindata.extend(bow_features(path(pos, i))); trainlabels.append(1)
        traindata.extend(bow_features(path(neg, i))); trainlabels.append(-1)

    print '- feature_extraction the model'
    svm = svm.NuSVC(nu=0.5, kernel='rbf', gamma=0.1, probability=True)
    svm.fit(np.array(traindata), np.array(trainlabels))
    joblib.dump(svm, CONST_SVM_MODEL_PATH, compress=3)
else:
    print '- loading the model'
    svm = joblib.load(CONST_SVM_MODEL_PATH)

predictions = []
tot_geral_pos, tot_geral_neg = 0, 0
tot_pos_pos, tot_pos_neg = 0, 0
tot_neg_pos, tot_neg_neg = 0, 0
#pos=0
print '- testing the model (pos) -> ' + CONST_TEST_DATA_POS_PATH
for file in os.listdir(CONST_TEST_DATA_POS_PATH):
    if file != '.DS_Store':
        tot_geral_pos += 1
        obj_predict = predict(CONST_TEST_DATA_POS_PATH + file)
        #if (obj_predict[1][0][0] == 1.0):
        if obj_predict[0] == 1.0:
            tot_pos_pos += 1
        if obj_predict[0] == -1.0:
            tot_pos_neg += 1
        #if 'pos-' in file:
        #    pos = pos + 1
        predictions.append([file, obj_predict[0], '-1'])

print '- testing the model (neg) -> ' + CONST_TEST_DATA_NEG_PATH
for file in os.listdir(CONST_TEST_DATA_NEG_PATH):
    if file != '.DS_Store':
        tot_geral_neg += 1
        obj_predict = predict(CONST_TEST_DATA_NEG_PATH + file)
        #if (obj_predict[1][0][0] == 1.0):
        if obj_predict[0] == 1.0:
            tot_neg_pos += 1
        if obj_predict[0] == -1.0:
            tot_neg_neg += 1
        #if 'pos-' in file:
        #    pos = pos + 1
        predictions.append([file, obj_predict[0], '-1'])

print '- saving predictions...'
predictions = np.array(predictions)
np.savetxt(CONST_PREDICTIONS, predictions, fmt='%s', delimiter=',', newline=os.linesep)

print 'tot files neg = %s | tot files pos = %s' % (tot_geral_neg, tot_geral_pos)
print 'tot files = ' + str(tot_geral_neg + tot_geral_pos)
print 'tot_pos_pos = ' + str(tot_pos_pos) + ' (' + "{0:.2f}".format(float(tot_pos_pos) / float(tot_geral_pos)) + ')'
print 'tot_pos_neg = ' + str(tot_pos_neg) + ' (' + "{0:.2f}".format(float(tot_pos_neg / float(tot_geral_pos))) + ')'
print 'tot_neg_neg = ' + str(tot_neg_neg) + ' (' + "{0:.2f}".format(float(tot_neg_neg) / float(tot_geral_neg)) + ')'
print 'tot_neg_pos = ' + str(tot_neg_pos) + ' (' + "{0:.2f}".format(float(tot_neg_pos / float(tot_geral_neg))) + ')'
acc = float(tot_pos_pos + tot_neg_neg) / float(tot_geral_neg + tot_geral_pos)
print 'acc = ' + str("{0:.2f}".format(acc))
print 'err = ' + str("{0:.2f}".format(float(1 - acc)))

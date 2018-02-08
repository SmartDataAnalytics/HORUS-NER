import cv2
from sklearn.externals import joblib

from horus.core.util.systemlog import SysLogger


class SIFT():
    def __init__(self, config):
        self.detect = cv2.xfeatures2d.SIFT_create()
        self.extract = cv2.xfeatures2d.SIFT_create()
        self.flann_params = dict(algorithm=1, trees=config.models_kmeans_trees)
        self.flann = cv2.FlannBasedMatcher(self.flann_params, {})
        self.extract_bow = cv2.BOWImgDescriptorExtractor(self.extract, self.flann)
        self.svm_logo = joblib.load(config.models_cv_org)
        self.voc_org = joblib.load(config.models_cv_org_dict)
        self.svm_loc1 = joblib.load(config.models_cv_loc1)
        self.svm_loc2 = joblib.load(config.models_cv_loc2)
        self.svm_loc3 = joblib.load(config.models_cv_loc3)
        self.svm_loc4 = joblib.load(config.models_cv_loc4)
        self.svm_loc5 = joblib.load(config.models_cv_loc5)
        self.svm_loc6 = joblib.load(config.models_cv_loc6)
        self.svm_loc7 = joblib.load(config.models_cv_loc7)
        self.svm_loc8 = joblib.load(config.models_cv_loc8)
        self.svm_loc9 = joblib.load(config.models_cv_loc9)
        self.svm_loc10 = joblib.load(config.models_cv_loc10)
        self.voc_loc_1 = joblib.load(config.models_cv_loc_1_dict)
        self.voc_loc_2 = joblib.load(config.models_cv_loc_2_dict)
        self.voc_loc_3 = joblib.load(config.models_cv_loc_3_dict)
        self.voc_loc_4 = joblib.load(config.models_cv_loc_4_dict)
        self.voc_loc_5 = joblib.load(config.models_cv_loc_5_dict)
        self.voc_loc_6 = joblib.load(config.models_cv_loc_6_dict)
        self.voc_loc_7 = joblib.load(config.models_cv_loc_7_dict)
        self.voc_loc_8 = joblib.load(config.models_cv_loc_8_dict)
        self.voc_loc_9 = joblib.load(config.models_cv_loc_9_dict)
        self.voc_loc_10 = joblib.load(config.models_cv_loc_10_dict)
        self.face_cascade = cv2.CascadeClassifier(config.models_cv_per)

    def bow_features(self, fn, ner_type):
        im = cv2.imread(fn, 0)
        if ner_type == 'ORG_1':
            self.extract_bow.setVocabulary(self.voc_org)
        elif ner_type == 'LOC_1':
            self.extract_bow.setVocabulary(self.voc_loc_1)
        elif ner_type == 'LOC_2':
            self.extract_bow.setVocabulary(self.voc_loc_2)
        elif ner_type == 'LOC_3':
            self.extract_bow.setVocabulary(self.voc_loc_3)
        elif ner_type == 'LOC_4':
            self.extract_bow.setVocabulary(self.voc_loc_4)
        elif ner_type == 'LOC_5':
            self.extract_bow.setVocabulary(self.voc_loc_5)
        elif ner_type == 'LOC_6':
            self.extract_bow.setVocabulary(self.voc_loc_6)
        elif ner_type == 'LOC_7':
            self.extract_bow.setVocabulary(self.voc_loc_7)
        elif ner_type == 'LOC_8':
            self.extract_bow.setVocabulary(self.voc_loc_8)
        elif ner_type == 'LOC_9':
            self.extract_bow.setVocabulary(self.voc_loc_9)
        elif ner_type == 'LOC_10':
            self.extract_bow.setVocabulary(self.voc_loc_10)

        return self.extract_bow.compute(im, self.detect.detect(im))

    def detect_logo(self, img):
        try:
            f = self.bow_features(img, 'ORG_1');
            if f is None:
                p = -1
            else:
                p = self.svm_logo.predict(f)[0]
                if p==-1: p=0
            return p

        except Exception as error:
            return 0

    def detect_place(self, img):
        try:
            ret = []
            f = self.bow_features(img, 'LOC_1');
            if f is None:
                ret.append(-1)
            else:
                y = self.svm_loc1.predict(f)[0]
                if y==-1:y=0
                ret.append(y)

            f = self.bow_features(img, 'LOC_2');
            if f is None:
                ret.append(-1)
            else:
                y = self.svm_loc2.predict(f)[0]
                if y==-1:y=0
                ret.append(y)

            f = self.bow_features(img, 'LOC_3');
            if f is None:
                ret.append(-1)
            else:
                y = self.svm_loc3.predict(f)[0]
                if y==-1:y=0
                ret.append(y)

            f = self.bow_features(img, 'LOC_4');
            if f is None:
                ret.append(-1)
            else:
                y = self.svm_loc4.predict(f)[0]
                if y==-1:y=0
                ret.append(y)

            f = self.bow_features(img, 'LOC_5');
            if f is None:
                ret.append(-1)
            else:
                y = self.svm_loc5.predict(f)[0]
                if y==-1:y=0
                ret.append(y)

            f = self.bow_features(img, 'LOC_6');
            if f is None:
                ret.append(-1)
            else:
                y = self.svm_loc6.predict(f)[0]
                if y==-1:y=0
                ret.append(y)

            f = self.bow_features(img, 'LOC_7');
            if f is None:
                ret.append(-1)
            else:
                y = self.svm_loc7.predict(f)[0]
                if y==-1:y=0
                ret.append(y)

            f = self.bow_features(img, 'LOC_8');
            if f is None:
                ret.append(-1)
            else:
                y = self.svm_loc8.predict(f)[0]
                if y==-1:y=0
                ret.append(y)

            f = self.bow_features(img, 'LOC_9');
            if f is None:
                ret.append(-1)
            else:
                y = self.svm_loc9.predict(f)[0]
                if y==-1:y=0
                ret.append(y)

            f = self.bow_features(img, 'LOC_10');
            if f is None:
                ret.append(-1)
            else:
                y = self.svm_loc10.predict(f)[0]
                if y==-1:y=0
                ret.append(y)

            return ret

        except:
            return [0] * 10

    def detect_faces(self, img):
        try:
            # print cv2.__version__
            image = cv2.imread(img)
            if image is None:
                raise('could not load the image: ' + img)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
            return len(faces)
            # cv2.CV_HAAR_SCALE_IMAGE #
            # minSize=(30, 30)

            ## Draw a rectangle around the faces
            # for (x, y, w, h) in faces:
            #    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.imshow("Faces found", image)
            # cv2.waitKey(0)

        except Exception as error:
            return 0
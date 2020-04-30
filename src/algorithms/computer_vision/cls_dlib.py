import dlib as dlib
from skimage import io
from config import HorusConfig


class DLib_Classifier():
    def __init__(self, config):
        try:
            self.config = config
            self.config.logger.debug('loading DLib')
            self.detector = dlib.get_frontal_face_detector()
            self.cnn_face_detector = dlib.cnn_face_detection_model_v1(config.dir_models + 'dlib/mmod_human_face_detector.dat')
        except Exception:
            raise

    def detect_faces_cnn(self, img_path):
        try:
            img = io.imread(img_path)
            return (len(self.cnn_face_detector(img, 1)))
        except Exception as e:
            self.config.logger.error(e)
            return 0


if __name__ == '__main__':
    pos_img_places = ['18_0_3', '18_0_4', '18_0_5', '18_0_9', '18_0_10', '20_0_1', '20_0_2', '20_0_3', '20_0_4',
                      '20_0_5',
                      '20_0_6', '20_0_7', '20_0_8', '20_0_9', '20_0_10', '38_0_1', '38_0_2', '38_0_3']
    pos_img_logos = ['28_0_1', '28_0_2', '28_0_3', '28_0_10', '68_0_8', '102_0_10', '104_0_8', '104_0_4', '114_0_8',
                     '134_0_4',
                     '188_0_1', '188_0_2', '188_0_4', '188_0_5', '198_0_2', '200_0_4']

    pos_img_per = ['4_0_3', '4_0_4', '4_0_5', '4_0_6', '4_0_7', '6_0_7', '6_0_6', '8_0_6', '6_0_8']
    config = HorusConfig()

    cls = DLib_Classifier(config)

    for img in pos_img_per:
        f = config.cache_img_folder + img + '.jpg'
        img = io.imread(f)
        dets = cls.detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))

        # dets, scores, idx = detector.run(img, 1, -1)
        # for i, d in enumerate(dets):
        #    print("Detection {}, score: {}, face_type:{}".format(
        #        d, scores[i], idx[i]))

        dets = cls.cnn_face_detector(img, 1)
        print("Number of CNN faces detected: {}".format(len(dets)))

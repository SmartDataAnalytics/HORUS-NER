from itertools import chain

import shorttext
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.slim.python.slim.nets import inception
from nltk.corpus import wordnet as wn

from src.classifiers.text_classification.topic_modeling_cnn import TopicModelingShortCNN
from src.classifiers.util.inception import dataset_utils, imagenet, inception_preprocessing
from src.config import HorusConfig
from src.core.util.systemlog import SysLogger


class InceptionCV():
    def __init__(self, config):
        try:
            self.config = config
            self.logger = SysLogger().getLog()
            self.DIR_MODELS = config.dir_models + "/inception/"
            self.TF_MODELS_URL = "http://download.tensorflow.org/models/"
            self.INCEPTION_V3_URL = self.TF_MODELS_URL + "inception_v3_2016_08_28.tar.gz"
            self.INCEPTION_V4_URL = self.TF_MODELS_URL + "inception_v4_2016_09_09.tar.gz"
            self.INCEPTION_V3_CKPT_PATH = self.DIR_MODELS + "inception_v3.ckpt"
            self.INCEPTION_V4_CKPT_PATH = self.DIR_MODELS + "inception_v4.ckpt"
            seed_PER = ['person', 'human being', 'man', 'woman', 'human body', 'human face']
            seed_ORG = ['logo', 'logotype']
            seed_LOC = ['volcano', 'stone', 'landscape', 'beach', 'sky', 'building', 'road', 'ocean', 'sea', 'lake',
                        'square']

            self.seeds = {'PER': seed_PER, 'ORG': seed_ORG, 'LOC': seed_LOC}

            if not tf.gfile.Exists(self.DIR_MODELS):
                tf.gfile.MakeDirs(self.DIR_MODELS)

            if not os.path.exists(self.INCEPTION_V3_CKPT_PATH):
                dataset_utils.download_and_uncompress_tarball(self.INCEPTION_V3_URL, self.DIR_MODELS)

            if not os.path.exists(self.INCEPTION_V4_CKPT_PATH):
                dataset_utils.download_and_uncompress_tarball(self.INCEPTION_V4_URL, self.DIR_MODELS)

        except Exception as e:
            raise e

    def process_image(self, image):
        filename = self.config.cache_img_folder + image
        with open(filename, "rb") as f:
            image_str = f.read()

        if image.endswith('jpg'):
            raw_image = tf.image.decode_jpeg(image_str, channels=3)
        elif image.endswith('png'):
            raw_image = tf.image.decode_png(image_str, channels=3)
        else:
            print("Image must be either jpg or png")
            return

        image_size = 299  # ImageNet image size, different models may be sized differently
        processed_image = inception_preprocessing.preprocess_image(raw_image, image_size,
                                                                   image_size, is_training=False)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            raw_image, processed_image = sess.run([raw_image, processed_image])

        return raw_image, processed_image.reshape(-1, 299, 299, 3)

    def plot_color_image(self, image):
        plt.figure(figsize=(10, 10))
        plt.imshow(image.astype(np.uint8), interpolation='nearest')
        plt.axis('off')

    def predict(self, image, version='V3', top=5):
        '''
        :param image: a path for an image
        :param version: inception's model version
        :return: top 10 predictions
        '''
        tf.reset_default_graph()

        # Process the image
        raw_image, processed_image = self.process_image(image)
        class_names = imagenet.create_readable_names_for_imagenet_labels()

        # Create a placeholder for the images
        X = tf.placeholder(tf.float32, [None, 299, 299, 3], name="X")

        '''
        inception_v3 function returns logits and end_points dictionary
        logits are output of the network before applying softmax activation
        '''

        if version.upper() == 'V3':
            model_ckpt_path = self.INCEPTION_V3_CKPT_PATH
            with arg_scope(inception.inception_v3_arg_scope()):
                # Set the number of classes and is_training parameter
                logits, end_points = inception.inception_v3(X, num_classes=1001, is_training=False)

        elif version.upper() == 'V4':
            model_ckpt_path = self.INCEPTION_V4_CKPT_PATH
            with arg_scope(inception.inception_v3_arg_scope()):
                # Set the number of classes and is_training parameter
                # Logits
                logits, end_points = inception.inception_v4(X, num_classes=1001, is_training=False)

        predictions = end_points.get('Predictions', 'No key named predictions')
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, model_ckpt_path)
            prediction_values = predictions.eval({X: processed_image})

        try:
            # Add an index to predictions and then sort by probability
            prediction_values = [(i, prediction) for i, prediction in enumerate(prediction_values[0, :])]
            prediction_values = sorted(prediction_values, key=lambda x: x[1], reverse=True)

            # Plot the image
            #self.plot_color_image(raw_image)
            #plt.show()
            #print("Using Inception_{} CNN\nPrediction: Probability\n".format(version))
            # Display the image and predictions
            #for i in range(top):
            #    predicted_class = class_names[prediction_values[i][0]]
            #    probability = prediction_values[i][1]
            #    print("{}: {:.2f}%".format(predicted_class, probability * 100))

            return prediction_values[0:top]

        # If the predictions do not come out right
        except:
            print(predictions)

    def detect_faces(self, image):
        try:
            out = self.predict(image, version='V4')
            print(out)
        except Exception as e:
            self.logger.error(e)
            return 0

    def detect_logos(self, image):
        try:
            out = self.predict(image, version='V4')
            print(out)
        except Exception as e:
            self.logger.error(e)
            return 0

    def detect_place(self, image):
        try:
            out = self.predict(image, version='V4')
            print(out)
        except Exception as e:
            self.logger.error(e)
            return 0

import logging
import os
from ConfigParser import SafeConfigParser

import datetime
import pkg_resources


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class HorusConfig(object):
    __metaclass__ = Singleton

    def __init__(self):
        fine = False

        self.logger = logging.getLogger('horus')

        for ini_file in os.path.dirname(os.path.abspath(__file__)) + '/../', os.curdir, os.path.expanduser("~"), "/etc/horus", os.environ.get('HORUS_CONF'):
            try:
                self.version = "0.2.0"
                self.version_label = "HORUS 0.2.0"
                self.description = "A framework to boost NLP tasks"

                with open(os.path.join(ini_file, "horus.ini")) as source:

                    #config.readfp(source)
                    parser = SafeConfigParser()
                    parser.read(source.name)

                    print(parser.get('conf', 'code'))
                    self.log_level = parser.get('conf', 'log_level')

                    #self.src_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
                    #self.root_dir = os.path.abspath(os.path.join(self.src_dir, os.pardir)) + '/'
                    self.root_dir = parser.get('path', 'root_dir')
                    self.root_dir_output = self.root_dir + 'data/'

                    #models_rootdir = pkg_resources.resource_filename('horus.resources', 'models') + "/"

                    # not attached to a root project directory, once it normally can be stored somewhere else (i.e., full path here)
                    #self.database_db = parser.get('path', 'database_path')
                    self.root_dir_tensorflow = parser.get('path', 'tensorflow_data')

                    # under \data folder
                    self.dir_encoders = self.root_dir_output + 'encoders/'
                    self.dir_log = self.root_dir_output + 'log/'
                    self.dir_models = self.root_dir_output + 'models/'
                    self.dir_output = self.root_dir_output + 'output/'
                    self.dir_datasets = self.root_dir_output + 'datasets/'
                    self.database_db  = self.root_dir_output + 'cache/microsoft_bing/horus.db'
                    self.cache_img_folder = self.root_dir_output + 'cache/microsoft_bing/img/'

                    self.nr_thread = parser.get('conf', 'nr_thread')

                    '''
                     ----------------------------------- Models -----------------------------------
                    '''

                    self.models_cnn_loc1 = self.dir_models + parser.get('models-cnn', 'horus_loc_1')
                    self.models_cnn_loc2 = self.dir_models + parser.get('models-cnn', 'horus_loc_2')
                    self.models_cnn_loc3 = self.dir_models + parser.get('models-cnn', 'horus_loc_3')
                    self.models_cnn_loc4 = self.dir_models + parser.get('models-cnn', 'horus_loc_4')
                    self.models_cnn_loc5 = self.dir_models + parser.get('models-cnn', 'horus_loc_5')
                    self.models_cnn_loc6 = self.dir_models + parser.get('models-cnn', 'horus_loc_6')
                    self.models_cnn_loc7 = self.dir_models + parser.get('models-cnn', 'horus_loc_7')
                    self.models_cnn_loc8 = self.dir_models + parser.get('models-cnn', 'horus_loc_8')
                    self.models_cnn_loc9 = self.dir_models + parser.get('models-cnn', 'horus_loc_9')
                    self.models_cnn_loc10 = self.dir_models + parser.get('models-cnn', 'horus_loc_10')
                    self.models_cnn_per = self.dir_models + parser.get('models-cnn', 'horus_per')
                    self.models_cnn_org = self.dir_models + parser.get('models-cnn', 'horus_org')

                    self.models_cv_loc1 = self.dir_models + parser.get('models-cv', 'horus_loc_1')
                    self.models_cv_loc2 = self.dir_models + parser.get('models-cv', 'horus_loc_2')
                    self.models_cv_loc3 = self.dir_models + parser.get('models-cv', 'horus_loc_3')
                    self.models_cv_loc4 = self.dir_models + parser.get('models-cv', 'horus_loc_4')
                    self.models_cv_loc5 = self.dir_models + parser.get('models-cv', 'horus_loc_5')
                    self.models_cv_loc6 = self.dir_models + parser.get('models-cv', 'horus_loc_6')
                    self.models_cv_loc7 = self.dir_models + parser.get('models-cv', 'horus_loc_7')
                    self.models_cv_loc8 = self.dir_models + parser.get('models-cv', 'horus_loc_8')
                    self.models_cv_loc9 = self.dir_models + parser.get('models-cv', 'horus_loc_9')
                    self.models_cv_loc10 = self.dir_models + parser.get('models-cv', 'horus_loc_10')

                    self.models_cv_loc_1_dict = self.dir_models + parser.get('models-cv', 'horus_loc_1_voc')
                    self.models_cv_loc_2_dict = self.dir_models + parser.get('models-cv', 'horus_loc_2_voc')
                    self.models_cv_loc_3_dict = self.dir_models + parser.get('models-cv', 'horus_loc_3_voc')
                    self.models_cv_loc_4_dict = self.dir_models + parser.get('models-cv', 'horus_loc_4_voc')
                    self.models_cv_loc_5_dict = self.dir_models + parser.get('models-cv', 'horus_loc_5_voc')
                    self.models_cv_loc_6_dict = self.dir_models + parser.get('models-cv', 'horus_loc_6_voc')
                    self.models_cv_loc_7_dict = self.dir_models + parser.get('models-cv', 'horus_loc_7_voc')
                    self.models_cv_loc_8_dict = self.dir_models + parser.get('models-cv', 'horus_loc_8_voc')
                    self.models_cv_loc_9_dict = self.dir_models + parser.get('models-cv', 'horus_loc_9_voc')
                    self.models_cv_loc_10_dict = self.dir_models + parser.get('models-cv', 'horus_loc_10_voc')

                    self.models_cv_org = self.dir_models + parser.get('models-cv', 'horus_org')
                    self.models_cv_org_dict = self.dir_models + parser.get('models-cv', 'horus_org_voc')
                    self.models_cv_per = self.dir_models + parser.get('models-cv', 'horus_per')

                    self.models_1_text = self.dir_models + parser.get('models-text', 'horus_textchecking_1')
                    self.models_2_text = self.dir_models + parser.get('models-text', 'horus_textchecking_2')
                    self.models_3_text = self.dir_models + parser.get('models-text', 'horus_textchecking_3')
                    self.models_4_text = self.dir_models + parser.get('models-text', 'horus_textchecking_4')
                    self.models_5_text = self.dir_models + parser.get('models-text', 'horus_textchecking_5')

                    self.models_1_text_cnn = self.dir_models + parser.get('models-text', 'horus_texthecking_tm_cnn')

                    self.model_final = self.dir_models + parser.get('models-horus', 'horus_final')
                    self.model_final_encoder = self.dir_models + parser.get('models-horus', 'horus_final_encoder')

                    self.model_stanford_filename_pos = self.dir_models + parser.get('model-stanford', 'model_filename_pos')
                    self.model_stanford_path_jar_pos = self.dir_models + parser.get('model-stanford', 'path_to_jar_pos')
                    self.model_stanford_filename_ner = self.dir_models + parser.get('model-stanford', 'model_filename_ner')
                    self.model_stanford_path_jar_ner = self.dir_models + parser.get('model-stanford', 'path_to_jar_ner')

                    self.models_tweetnlp_jar = self.dir_models + parser.get('models-tweetnlp', 'path_to_jar_pos')
                    self.models_tweetnlp_model = self.dir_models + parser.get('models-tweetnlp', 'model_filename_pos')
                    self.models_tweetnlp_java_param = parser.get('models-tweetnlp', 'java_param')

                    self.search_engine_api = parser.get('search-engine', 'api')
                    self.search_engine_key = parser.get('search-engine', 'key')
                    self.search_engine_features_text = parser.get('search-engine', 'features_text')
                    self.search_engine_features_img = parser.get('search-engine', 'features_img')
                    self.search_engine_tot_resources = parser.get('search-engine', 'tot_resources')


                    self.translation_id = parser.get('translation', 'microsoft_client_id')
                    self.translation_secret = parser.get('translation', 'microsoft_client_secret')


                    self.cache_sentences = parser.get('cache', 'cache_sentences')

                    self.models_force_download = parser.get('models-param', 'force_download')
                    self.models_location_theta = parser.get('models-param', 'location_theta')
                    self.models_distance_theta = parser.get('models-param', 'distance_theta')
                    self.models_safe_interval = parser.get('models-param', 'safe_interval')
                    self.models_limit_min_loc = parser.get('models-param', 'limit_min_loc')
                    self.models_distance_theta_high_bias = parser.get('models-param', 'distance_theta_high_bias')
                    self.models_pos_tag_lib = int(parser.get('models-param', 'pos_tag_lib'))
                    self.models_pos_tag_lib_type = int(parser.get('models-param', 'pos_tag_lib_type'))
                    self.models_kmeans_trees = int(parser.get('models-param', 'kmeans-trees'))
                    self.object_detection_type = int(parser.get('models-param', 'object_detection_type'))
                    self.text_classification_type = int(parser.get('models-param', 'text_classification_type'))
                    self.embeddings_path = parser.get('models-param', 'embeddings_path')

                    self.mod_text_tfidf_active = int(parser.get('rest-interface', 'mod_text_tfidf_active'))
                    self.mod_text_topic_active = int(parser.get('rest-interface', 'mod_text_topic_active'))
                    self.mod_image_sift_active = int(parser.get('rest-interface', 'mod_image_sift_active'))
                    self.mod_image_cnn_active = int(parser.get('rest-interface', 'mod_image_cnn_active'))

                    fine = True

                    break
                    #config.readfp(source)

            except IOError:
                pass

        if fine is False:
            raise ValueError('error on trying to read the conf file (horus.conf)! Please set HORUS_CONF with its '
                             'path or place it at your home dir')
        else:
            if len(self.logger.handlers) == 0:
                self.logger.setLevel(logging.DEBUG)
                if self.log_level=='INFO':
                    self.logger.setLevel(logging.INFO)
                elif self.log_level=='WARNING':
                    self.logger.setLevel(logging.WARNING)
                elif self.log_level=='ERROR':
                    self.logger.setLevel(logging.ERROR)
                elif self.log_level=='CRITICAL':
                    self.logger.setLevel(logging.CRITICAL)

                now = datetime.datetime.now()
                handler = logging.FileHandler(self.dir_log + 'horus_' + now.strftime("%Y-%m-%d") + '.log')
                formatter = logging.Formatter(
                    "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                consoleHandler = logging.StreamHandler()
                consoleHandler.setFormatter(formatter)
                self.logger.addHandler(consoleHandler)

        self.logger.info('==================================================================')
        self.logger.info('HORUS Framework')
        self.logger.info('version: ' + self.version)
        self.logger.info('more info: http://horus-ner.org/')
        self.logger.info('==================================================================')
        #ini_file = pkg_resources.resource_filename('resource', "horus.conf")
        #rootdir = os.getcwd()
        #

    @staticmethod
    def get_report():
        return 'to be implemented'
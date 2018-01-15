import os
from ConfigParser import SafeConfigParser
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
        config = None
        for ini_file in os.curdir, os.path.expanduser("~"), "/etc/horus", os.environ.get('HORUS_CONF'):
            try:
                with open(os.path.join(ini_file, "horus.ini")) as source:

                    #config.readfp(source)
                    parser = SafeConfigParser()
                    parser.read(source.name)

                    #self.root_dir = os.path.dirname(os.path.abspath(__file__))
                    self.root_dir = parser.get('path', 'root_dir')
                    models_rootdir = pkg_resources.resource_filename('horus.resources', 'models') + "/"

                    # not attached to a root project directory, once it normally can be stored somewhere else (i.e., full path here)
                    self.database_db = parser.get('path', 'database_path')

                    self.output_path = self.root_dir + parser.get('path', 'output_path')
                    self.dataset_path = self.root_dir + parser.get('path', 'dataset_path')
                    self.experiment_path = self.root_dir + parser.get('path', 'experiment_path')


                    self.encoder_path = models_rootdir + parser.get('path', 'encoder_path')

                    self.models_cv_loc1 = models_rootdir + parser.get('models-cv', 'horus_loc_1')
                    self.models_cv_loc2 = models_rootdir + parser.get('models-cv', 'horus_loc_2')
                    self.models_cv_loc3 = models_rootdir + parser.get('models-cv', 'horus_loc_3')
                    self.models_cv_loc4 = models_rootdir + parser.get('models-cv', 'horus_loc_4')
                    self.models_cv_loc5 = models_rootdir + parser.get('models-cv', 'horus_loc_5')
                    self.models_cv_loc6 = models_rootdir + parser.get('models-cv', 'horus_loc_6')
                    self.models_cv_loc7 = models_rootdir + parser.get('models-cv', 'horus_loc_7')
                    self.models_cv_loc8 = models_rootdir + parser.get('models-cv', 'horus_loc_8')
                    self.models_cv_loc9 = models_rootdir + parser.get('models-cv', 'horus_loc_9')
                    self.models_cv_loc10 = models_rootdir + parser.get('models-cv', 'horus_loc_10')

                    self.models_cv_loc_1_dict = models_rootdir + parser.get('models-cv', 'horus_loc_1_voc')
                    self.models_cv_loc_2_dict = models_rootdir + parser.get('models-cv', 'horus_loc_2_voc')
                    self.models_cv_loc_3_dict = models_rootdir + parser.get('models-cv', 'horus_loc_3_voc')
                    self.models_cv_loc_4_dict = models_rootdir + parser.get('models-cv', 'horus_loc_4_voc')
                    self.models_cv_loc_5_dict = models_rootdir + parser.get('models-cv', 'horus_loc_5_voc')
                    self.models_cv_loc_6_dict = models_rootdir + parser.get('models-cv', 'horus_loc_6_voc')
                    self.models_cv_loc_7_dict = models_rootdir + parser.get('models-cv', 'horus_loc_7_voc')
                    self.models_cv_loc_8_dict = models_rootdir + parser.get('models-cv', 'horus_loc_8_voc')
                    self.models_cv_loc_9_dict = models_rootdir + parser.get('models-cv', 'horus_loc_9_voc')
                    self.models_cv_loc_10_dict = models_rootdir + parser.get('models-cv', 'horus_loc_10_voc')

                    self.models_cv_org = models_rootdir + parser.get('models-cv', 'horus_org')
                    self.models_cv_org_dict = models_rootdir + parser.get('models-cv', 'horus_org_voc')
                    self.models_cv_per = models_rootdir + parser.get('models-cv', 'horus_per')

                    self.models_text_root =  models_rootdir + parser.get('models-text', 'root')
                    self.models_1_text = models_rootdir + parser.get('models-text', 'horus_textchecking_1')
                    self.models_2_text = models_rootdir + parser.get('models-text', 'horus_textchecking_2')
                    self.models_3_text = models_rootdir + parser.get('models-text', 'horus_textchecking_3')
                    self.models_4_text = models_rootdir + parser.get('models-text', 'horus_textchecking_4')
                    self.models_5_text = models_rootdir + parser.get('models-text', 'horus_textchecking_5')

                    self.model_final = models_rootdir + parser.get('models-horus', 'horus_final')
                    self.model_final_encoder = models_rootdir + parser.get('models-horus', 'horus_final_encoder')

                    self.model_stanford_filename_pos = models_rootdir + parser.get('model-stanford', 'model_filename_pos')
                    self.model_stanford_path_jar_pos = models_rootdir + parser.get('model-stanford', 'path_to_jar_pos')
                    self.model_stanford_filename_ner = models_rootdir + parser.get('model-stanford', 'model_filename_ner')
                    self.model_stanford_path_jar_ner = models_rootdir + parser.get('model-stanford', 'path_to_jar_ner')

                    self.models_tweetnlp_jar = models_rootdir + parser.get('models-tweetnlp', 'path_to_jar_pos')
                    self.models_tweetnlp_model = models_rootdir + parser.get('models-tweetnlp', 'model_filename_pos')
                    self.models_tweetnlp_java_param = parser.get('models-tweetnlp', 'java_param')

                    self.search_engine_api = parser.get('search-engine', 'api')
                    self.search_engine_key = parser.get('search-engine', 'key')
                    self.search_engine_features_text = parser.get('search-engine', 'features_text')
                    self.search_engine_features_img = parser.get('search-engine', 'features_img')
                    self.search_engine_tot_resources = parser.get('search-engine', 'tot_resources')


                    self.translation_id = parser.get('translation', 'microsoft_client_id')
                    self.translation_secret = parser.get('translation', 'microsoft_client_secret')

                    self.cache_img_folder = parser.get('cache', 'img_folder')

                    self.dataset_ritter = parser.get('dataset', 'ds_ritter')
                    self.dataset_conll = parser.get('dataset', 'ds_conll')

                    self.models_force_download = parser.get('models-param', 'force_download')
                    self.models_location_theta = parser.get('models-param', 'location_theta')
                    self.models_distance_theta = parser.get('models-param', 'distance_theta')
                    self.models_safe_interval = parser.get('models-param', 'safe_interval')
                    self.models_limit_min_loc = parser.get('models-param', 'limit_min_loc')
                    self.models_distance_theta_high_bias = parser.get('models-param', 'distance_theta_high_bias')
                    self.models_pos_tag_lib = int(parser.get('models-param', 'pos_tag_lib'))
                    self.models_pos_tag_lib_type = int(parser.get('models-param', 'pos_tag_lib_type'))
                    self.models_kmeans_trees = int(parser.get('models-param', 'kmeans-trees'))

                    fine = True

                    break
                    #config.readfp(source)
            except IOError:
                pass

        if fine is False:
            raise ValueError('error on trying to read the conf file (horus.conf)! Please set HORUS_CONF with its '
                             'path or place it at your home dir')

        #ini_file = pkg_resources.resource_filename('resource', "horus.conf")
        #rootdir = os.getcwd()
        #

    @staticmethod
    def get_report():
        return 'to be implemented'
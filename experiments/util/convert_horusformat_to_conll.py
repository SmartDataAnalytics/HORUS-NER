from horus.core.config import HorusConfig
from horus.util.data_conversion import convertHORUStoCoNLL

config = HorusConfig()

features = [3,4,5,6,11,12,13,14,15,16,17,19,20,21,22,24,25]
convertHORUStoCoNLL(config.output_path + "/experiments/EXP_do_tokenization/out_exp003_coNLL2003testA_en_NLTK.csv", features,
                    config.output_path + "/experiments/EXP_do_tokenization/conversion/out_exp003_coNLL2003testA_en_NLTK.csv")

convertHORUStoCoNLL(config.output_path + "/experiments/EXP_do_tokenization/out_exp003_ritter_en_tweetNLP.csv", features,
                    config.output_path + "/experiments/EXP_do_tokenization/conversion/out_exp003_ritter_en_tweetNLP.csv")

convertHORUStoCoNLL(config.output_path + "/experiments/EXP_do_tokenization/out_exp003_wnut15_en_tweetNLP.csv", features,
                    config.output_path + "/experiments/EXP_do_tokenization/conversion/out_exp003_wnut15_en_tweetNLP.csv")

convertHORUStoCoNLL(config.output_path + "/experiments/EXP_do_tokenization/out_exp003_wnut16_en_tweetNLP.csv", features,
                    config.output_path + "/experiments/EXP_do_tokenization/conversion/out_exp003_wnut16_en_tweetNLP.csv")
#if __name__ == "__main__":
#    if len(sys.argv) != 2:
#        print "please inform: 1: data set and 2: column indexes ([1, .., n])"
#    else:
#        convertHORUStoCoNLL(sys.argv[0], sys.argv[1])

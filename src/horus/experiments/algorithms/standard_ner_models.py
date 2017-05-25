from horus import definitions
from horus.components.config import HorusConfig
import pandas as pd
import re
import numpy
from sklearn.externals import joblib

"""
==========================================================
Experiments: 
    NER models X NER + HORUS features
==========================================================
Within this experiments we show the performance of standard
NER algorithms (Stanford NER and NLTK NER) with and without
using HORUS as features

input: 
- horus matrix file: see README -> https://github.com/diegoesteves/horus-ner

output:
- performance measures
"""
config = HorusConfig()
X, Y = [], []

def encode_features(features):
    if config.models_pos_tag_lib == 1:
        le = joblib.load(config.encoder_path + "encoder_nltk.pkl")
    elif config.models_pos_tag_lib == 2:
        le = joblib.load(config.encoder_path + "encoder_stanford.pkl")
    elif config.models_pos_tag_lib == 3:
        le = joblib.load(config.encoder_path + "encoder_tweetnlp.pkl")

    for x in features:
        x[0] = le.transform(x[0])
        x[1] = le.transform(x[1])
        x[2] = le.transform(x[2])
    return features

def get_features_from_horus_matrix(horusfile):
    features = []
    teste = []
    df = pd.read_csv(horusfile, delimiter=",", skiprows=1, header=None, keep_default_na=False, na_values=['_|_'])
    countaux = 0
    for index, linha in df.iterrows():
        countaux+=1
        if len(linha)>0:
            if linha[7] == 0:

                idsent = linha[1]
                idtoken = idsent = linha[2]

                pos_bef = ''
                pos_aft = ''
                if index > 0 and df.get_value(index-1,7) == 0:
                    pos_bef = df.get_value(index-1,5)
                if index + 1 < len(df) and df.get_value(index+1,7) == 0:
                    pos_aft = df.get_value(index+1,5)

                if linha[51] in definitions.NER_TAGS_LOC: Y.append(1)
                elif linha[51] in definitions.NER_TAGS_ORG: Y.append(2)
                elif linha[51] in definitions.NER_TAGS_PER: Y.append(3)
                else: Y.append(4)

                token = linha[3]
                pos = linha[5]
                one_char_token = 1 if len(token) == 1 else 0
                special_char = 1 if len(re.findall('(http://\S+|\S*[^\w\s]\S*)', token))>0 else 0
                first_capitalized = 1 if token[0].isupper() else 0
                capitalized = 1 if token.isupper() else 0
                title = 1 if token.istitle() else 0
                digit = 1 if token.isdigit() else 0
                nr_images_returned = linha[17]
                nr_websites_returned = linha[25]
                hyphen = 1 if '-' in token else 0
                cv_loc = int(linha[12])
                cv_org = int(linha[13])
                cv_per = int(linha[14])
                cv_dist = int(linha[15])
                cv_plc = int(linha[16])
                tx_loc = int(linha[20])
                tx_org = int(linha[21])
                tx_per = int(linha[22])
                tx_err = float(linha[23])
                tx_dist = float(linha[24])

                teste.append(linha[6])
                if linha[6] in definitions.NER_TAGS_LOC: ner = definitions.KLASSES2["LOC"]
                elif linha[6] in definitions.NER_TAGS_ORG: ner = definitions.KLASSES2["ORG"]
                elif linha[6] in definitions.NER_TAGS_PER: ner = definitions.KLASSES2["PER"]
                else: ner = definitions.KLASSES2["O"]

                features.append((idsent, idtoken, pos_bef, pos, pos_aft, title, digit,
                                 one_char_token, special_char, first_capitalized, hyphen,
                                 capitalized, nr_images_returned,
                                 cv_org, cv_loc, cv_per, cv_dist, cv_plc,
                                 tx_org, tx_loc, tx_per, tx_dist, tx_err))
    print len(Y)
    print set(Y)
    print set(teste)

    features = numpy.array(features)
    return features



f = config.output_path + "experiments/EXP_do_tokenization/out_exp003_ritter_en_tweetNLP.csv"
features = get_features_from_horus_matrix(f)
features = encode_features(features)
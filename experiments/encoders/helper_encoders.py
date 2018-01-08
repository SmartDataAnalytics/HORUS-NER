import numpy
from sklearn import preprocessing
from sklearn.externals import joblib

from horus.core.config import HorusConfig
from horus.core.nlp_tools import NLPTools


def extract_all_tags_conll(config):
    dspath = config.dataset_path + "coNLL2003/coNLL2003.eng.testa"
    try:
        postags_nltk, postags_nltk_u, gs = [''], [''], ['']
        s = ''
        isent = 0
        tools = NLPTools()
        with open(dspath) as f:
            for line in f:
                if line.strip() != '':
                    token = line.split()[0]
                    gs.append(line.split()[1])
                if line.strip() == '':  # that' a sentence
                    print isent
                    _tokens_nltk, _pos_nltk, _pos_uni_nltk = tools.tokenize_and_pos_nltk(s)
                    p1 = numpy.array(_pos_nltk)
                    p2 = numpy.array(_pos_uni_nltk)
                    if len(p1) > 0:
                        temp = numpy.concatenate((p1[:, 1], postags_nltk), axis=0)
                        postags_nltk = numpy.unique(temp)

                    if len(p2) > 0:
                        temp = numpy.concatenate((p2[:, 1], postags_nltk_u), axis=0)
                        postags_nltk_u = numpy.unique(temp)
                    s = ''
                    isent += 1

                else:
                    s += token + ' '
        return postags_nltk, postags_nltk_u, gs
    except Exception as error:
        print('caught this error: ' + repr(error))
        exit(-1)


def extract_pos_dictionary_from_conll(config):
    dspath = config.dataset_path + "Ritter/ner.txt"
    postags_nltk, postags_stanford, postags_tweet = [''], [''], ['']
    postags_nltk_u, postags_stanford_u, postags_tweet_u = [''], [''], ['']
    try:
        s = ''
        isent = 0
        tools = NLPTools()
        with open(dspath) as f:
            for line in f:
                if isent > 10000:
                    break
                if line.strip() != '':
                    token = line.split('\t')[0]
                if line.strip() == '':# that' a sentence
                    print isent

                    _tokens_nltk, _pos_nltk, _pos_uni_nltk = tools.tokenize_and_pos_nltk(s)
                    _tokens_st, _pos_st, _pos_uni_st = tools.tokenize_and_pos_stanford(s)
                    _tokens_twe, _pos_twe, _pos_uni_twe = tools.tokenize_and_pos_twitter(s)

                    p1 = numpy.array(_pos_nltk)
                    p2 = numpy.array(_pos_uni_nltk)
                    p3 = numpy.array(_pos_st)
                    p4 = numpy.array(_pos_uni_st)
                    p5 = numpy.array(_pos_twe)
                    p6 = numpy.array(_pos_uni_twe)

                    if len(p1) > 0:
                        temp = numpy.concatenate((p1[:,1], postags_nltk), axis=0)
                        postags_nltk = numpy.unique(temp)

                    if len(p2) > 0:
                        temp = numpy.concatenate((p2[:, 1], postags_nltk_u), axis=0)
                        postags_nltk_u = numpy.unique(temp)

                    if len(p3) > 0:
                        temp = numpy.concatenate((p3[:, 1], postags_stanford), axis=0)
                        postags_stanford = numpy.unique(temp)

                    if len(p4) > 0:
                        temp = numpy.concatenate((p4[:, 1], postags_stanford_u), axis=0)
                        postags_stanford_u = numpy.unique(temp)

                    if len(p5) > 0:
                        temp = numpy.concatenate((p5[:, 1], postags_tweet), axis=0)
                        postags_tweet = numpy.unique(temp)

                    if len(p6) > 0:
                        temp = numpy.concatenate((p6[:, 1], postags_tweet_u), axis=0)
                        postags_tweet_u = numpy.unique(temp)


                    '''
                    for p1 in _pos_nltk:
                        if p1[1] not in postags_nltk: postags_nltk.append(p1[1])
                    for p11 in _pos_uni_nltk:
                        if p11[1] not in postags_nltk_u: postags_nltk_u.append(p11[1])
                    for p2 in _pos_st:
                        if p2[1] not in postags_stanford: postags_stanford.append(p2[1])
                    for p22 in _pos_uni_st:
                        if p22[1] not in postags_stanford_u: postags_stanford_u.append(p22[1])
                    for p3 in _pos_twe:
                        if p3[1] not in postags_tweet: postags_tweet.append(p3[1])
                    for p33 in _pos_uni_twe:
                        if p33[1] not in postags_tweet_u: postags_tweet_u.append(p33[1])
                    '''
                    s = ''
                    isent += 1
                else:
                    s += token + ' '

        return postags_nltk, postags_nltk_u, postags_stanford, postags_stanford_u, postags_tweet, postags_tweet_u

    except Exception as error:
        print('caught this error: ' + repr(error))
        exit(-1)

config = HorusConfig()

n1, n2, n3 = extract_all_tags_conll(config)
all = numpy.append(numpy.array([]), n1)
all = numpy.append(all, n2)
all = numpy.append(all, n3)
allu = numpy.unique(all)
le = preprocessing.LabelEncoder()
le.fit(allu)
joblib.dump(le, config.encoder_path + '_encoder_nltk2.pkl', compress=3)
exit(0)

n1, n2, s1, s2, t1, t2 = extract_pos_dictionary_from_conll(config)


all = numpy.append(numpy.array([]), n1)
all = numpy.append(all, n2)
all = numpy.append(all, s1)
all = numpy.append(all, s2)
all = numpy.append(all, t1)
all = numpy.append(all, t2)
allu = numpy.unique(all)

le = preprocessing.LabelEncoder()

le.fit(allu)
joblib.dump(le, config.encoder_path + '_encoder_pos.pkl', compress=3)
print '-- ALL --'
print len(allu)
print (allu)

le.fit(n1)
joblib.dump(le, config.encoder_path + '_encoder_nltk.pkl', compress=3)
print '-- NLTK --'
print len(n1)
print (n1)

le.fit(n2)
joblib.dump(le, config.encoder_path + '_encoder_nltk_universal.pkl', compress=3)
print '-- NLTK universal --'
print len(n2)
print (n2)

le.fit(s1)
joblib.dump(le, config.encoder_path + '_encoder_stanford.pkl', compress=3)
print '-- Stanford --'
print len(s1)
print (s1)

le.fit(s2)
joblib.dump(le, config.encoder_path + '_encoder_stanford_universal.pkl', compress=3)
print '-- Stanford universal --'
print len(s2)
print (s2)

le.fit(t1)
joblib.dump(le, config.encoder_path + '_encoder_tweetnlp.pkl', compress=3)
print '-- TweetNLP --'
print len(t1)
print (t1)

le.fit(t2)
joblib.dump(le, config.encoder_path + '_encoder_tweetnlp_universal.pkl', compress=3)
print '-- TweetNLP universal --'
print len(t2)
print (t2)


'''

'''
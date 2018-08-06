import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import TweetTokenizer
import pprint
from sklearn.externals import joblib
import copy

lancaster_stemmer = LancasterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
tknzr = TweetTokenizer(preserve_case=True, strip_handles=False, reduce_len=False)
stop = set(stopwords.words('english'))

def get_tuples(dspath):
    sentences = []
    s = ''
    tokens = []
    ners = []
    poss = []
    tot_sentences = 0
    tot_toks = 0
    ners_by_position = []
    index = 0
    with open(dspath) as f:
        for line in f:
            if line.strip() != '':
                token = line.split('\t')[0].decode('utf-8')
                ner = line.split('\t')[1].replace('\r', '').replace('\n', '').decode('utf-8')
                '''
                if ner in definitions.NER_TAGS_ORG:
                    ner = 'ORG'
                elif ner in definitions.NER_TAGS_LOC:
                    ner = 'LOC'
                elif ner in definitions.NER_TAGS_PER:
                    ner = 'PER'
                else :
                    ner = 'O'
                '''
                #ners_by_position.append([index, len(token), ner])
                index += len(token) + 1
            if line.strip() == '':
                if len(tokens) != 0:
                    #poss = [x[1].decode('utf-8') for x in nltk.pos_tag(nltk.word_tokenize(s[:-1]))]
                    poss = [x[1].decode('utf-8') for x in nltk.pos_tag(tknzr.tokenize(s[:-1]))]


                    #if len(poss) == len(tokens): # tokenization doesnt affect position of NERs, i.e., same tokenization
                    sentences.append(zip(tokens, poss, ners))
                    #else:
                    #    aux = 0
                    #    for i in range(len()):
                    #        if aux <= tokens[i]

                    tokens = []
                    ners = []
                    s = ''
                    tot_sentences += 1
            else:
                s += token + ' '
                if token != 'O':
                    tot_toks += 1
                tokens.append(token)
                ners.append(ner)

    return sentences, tot_sentences, tot_toks


def sent2labels(sent):
    return [label.encode("utf-8") for token, postag, label in sent]

pp = pprint.PrettyPrinter(depth=6)

def stat_report(dataset_path):
    company = 0
    facility = 0
    geo_loc = 0
    movie = 0
    musicartist = 0
    other = 0
    person = 0
    product = 0
    sportsteam = 0
    tvshow = 0

    dataset, tot_sentences, tot_tokens = get_tuples(dataset_path)

    labels = [sent2labels(s) for s in dataset]

    for string in labels:
        for tok in string:
            if tok.find("B-company") != -1:
                company += 1

            if tok.find("B-facility") != -1:
                facility += 1

            if tok.find("B-geo-loc") != -1:
                geo_loc += 1

            if tok.find("B-movie") != -1:
                movie += 1

            if tok.find("B-musicartist") != -1:
                musicartist += 1

            if tok.find("B-other") != -1:
                other += 1

            if tok.find("B-person") != -1:
                person += 1

            if tok.find("B-product") != -1:
                product += 1

            if tok.find("B-sportsteam") != -1:
                sportsteam += 1

            if tok.find("B-tvshow") != -1:
                tvshow += 1

            res = {
                    'company' : company,
                    'facility' : facility,
                    'geo_loc':geo_loc,
                    'movie': movie,
                    'musicartist': musicartist,
                    'other': other,
                    'person':person,
                    'product':product,
                    'sportsteam':sportsteam,
                    'tvshow':tvshow,
                    'tot_sents': tot_sentences,
                    'tot_toks': tot_tokens,
                }
    pp.pprint(res)
    return res

def stat_report_17(dataset_path):
    person = 0
    location = 0
    corporation = 0
    product = 0
    creative_work = 0
    group = 0

    dataset, tot_sentences, tot_tokens = get_tuples(dataset_path)

    labels = [sent2labels(s) for s in dataset]

    for string in labels:
        for tok in string:
            if tok.find("B-person") != -1:
                person += 1

            if tok.find("B-location") != -1:
                location += 1

            if tok.find("B-corporation") != -1:
                corporation += 1

            if tok.find("B-product") != -1:
                product += 1

            if tok.find("B-creative-work") != -1:
                creative_work += 1

            if tok.find("B-group") != -1:
                group += 1
            res = {
                    'person' : person,
                    'location' : location,
                    'corporation':corporation,
                    'product': product,
                    'creative-work': creative_work,
                    'group': group,
                    'tot_sents': tot_sentences,
                    'tot_toks': tot_tokens,
                }
    pp.pprint(res)
    return res


def simialr(sent1, sent2):
    sim = 0
    for tok1 in sent1:
        for tok2 in sent2:
            if tok1 == tok2:
                sim += 1
    if abs(sim - len(sent1)) < 2 and abs(sim - len(sent2)) < 2:
        return True
    return False

# print ("train")
# stat_report('../test_data/WNUT/16/2016.conll.freebase') #test
# print("-----------------------------------------")
# # print("test")
# stat_report('../test_data/WNUT/16/train.txt') # train
# stat_report('../test_data/WNUT/16/test.txt') # test
# print("-----------------------------------------")
# res = stat_report('../test_data/WNUT/15/2015.conll.freebase')
# sum = 0
# for elem in res:
#     if elem != 'tot_sents' and elem != 'tot_toks':
#         sum+= res[elem]
#
# print sumx
#
# exit(0)
# stat_report('../test_data/ritter_ner.tsv')
# stat_report('../test_data/WNUT/16/2016.conll.freebase')
# stat_report('../test_data/WNUT/16/test.txt')
# exit(0)
# stat_report_17('../test_data/WNUT/17/wnut17train.conll') # train
# stat_report_17('../test_data/WNUT/17/emerging.test.annotated') # train


dataset_wnut16_train, X, Y = get_tuples('../test_data/WNUT/16/train.txt')
clean_tweets, X, Y = get_tuples('../test_data/WNUT/16/train.txt')
# dataset_wnut16_test = get_tuples('../../../data/test_data/WNUT/16/test.txt')
dataset_wnut17_train, X, Y = get_tuples('../test_data/WNUT/17/wnut17train.conll')
# dataset_wnut17_test = get_tuples('../../../data/test_data/WNUT/17/emerging.test.annotated')
dataset_ritters_train, X, Y = get_tuples('../test_data/ritter_ner.tsv')

print len(dataset_ritters_train), len(dataset_wnut16_train), len(clean_tweets)

for sent1 in dataset_ritters_train:
    sim = False
    found  = False
    for sent2 in dataset_wnut16_train:
        sim = simialr(sent1, sent2)
        if sim == True:
            found = True
    if found == False:
        clean_tweets.append(sent1)

joblib.dump(clean_tweets, 'clean_tweets_16_ritters.pkl', compress=9)
exit(0)
# joblib.dump(clean_tweets, 'clean_tweets.pkl', compress=9)
#
# clean_tweets_2 = joblib.load('clean_tweets.pkl')
#
#
# for sent1 in dataset_wnut17_train:
#     sim = False
#     found  = False
#     for sent2 in clean_tweets:
#         sim = simialr(sent1, sent2)
#         if sim == True:
#             found = True
#     if found == False:
#         clean_tweets_2.append(sent1)
#
# joblib.dump(clean_tweets_2, 'clean_tweets_16_17_ritters.pkl', compress=9)

tot_len = len(dataset_ritters_train) + len(dataset_wnut16_train) + len(dataset_wnut17_train)
print tot_len, len(dataset_ritters_train), len(dataset_wnut16_train), len(dataset_wnut17_train), len(clean_tweets_2)

exit(0)
cnt = 0
rem = []


# exit(0)
print cnt, len(dataset_ritters_train)
for sent1 in dataset_ritters_train:
    for sent2 in dataset_wnut17_train:
        sim = 0
        for tok1 in sent1:
            for tok2 in sent2:
                if tok1 == tok2:
                    sim+=1
        if abs(sim - len(sent1)) < 2 and abs(sim - len(sent2)) < 2:
            cnt+=1
            rem.append(sent1)

print cnt, len(dataset_ritters_train)
cnt = 0
for sent1 in dataset_ritters_train:
    for sent2 in dataset_wnut16_train:
        if sent1 == sent2:
            rem.append(sent1)
            cnt+=1
            # break
print cnt, len(dataset_ritters_train)

for sent1 in dataset_ritters_train:
    for sent2 in dataset_wnut17_train:
        if sent1 == sent2:
            rem.append(sent1)
            cnt+=1
            # break

print cnt, len(dataset_ritters_train)

for elem in rem:
    if elem in dataset_ritters_train:
        dataset_ritters_train.remove(elem)

print cnt, len(dataset_ritters_train)



# dataset_wnut15_train = get_tuples('../../../data/test_data/WNUT/15/2015.conll.freebase')
# dataset_wnut16_train = get_tuples('../test_data/WNUT/16/train.txt')
# dataset_wnut16_test = get_tuples('../test_data/WNUT/16/test.txt')
# dataset_wnut17_train = get_tuples('../../../data/test_data/WNUT/17/wnut17train.conll')
# dataset_ritters_train = get_tuples('../../../data/test_data/ritter_ner.tsv')

# train_sents = dataset_wnut16_train
#
#
# test_sents = dataset_wnut16_test
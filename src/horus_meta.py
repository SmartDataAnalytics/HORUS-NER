import json
import jsonpickle
from src.definitions import PRE_PROCESSING_STATUS


class WordFeaturesInterface:

    @staticmethod
    def get_key_by_value(dictionary: dict, feature_alias: str):
        for key, value in dictionary.items():
            if feature_alias == value:
                return key
        # this is a hammer because we keep the postag in the horus token object instead of in the lexical features set.
        if feature_alias == 'postag':
            return 'postag'
        raise Exception('feature not found! check the feature dict', feature_alias)

    @staticmethod
    def get_visual() -> dict:
        features = {
            0: 'blah'
        }
        reversed_features = dict([(value, key) for key, value in features.items()])

        return features, reversed_features

    @staticmethod
    def get_textual() -> dict:

        features = {
            0: 'total.results.search_engine',
            1: 'total.retrieved.results.search_engine',
            2: 'total.error.translation',
            3: 'avg.ovr.k.loc',
            4: 'avg.ovr.k.org',
            5: 'avg.ovr.k.per',
            6: 'avg.ovr.k.other',
            7: 'avg.probs1.k.loc',
            8: 'avg.probs1.k.org',
            9: 'avg.probs1.k.per',
            10: 'avg.probs1.k.other',
            11: 'avg.probs2.k.loc',
            12: 'avg.probs2.k.org',
            13: 'avg.probs2.k.per',
            14: 'avg.probs2.k.other',
            15: 'top.binary.k',
            16: 'dist.k',
            17: 'total.topic.k.loc',
            18: 'total.topic.k.org',
            19: 'total.topic.k.per',
            20: 'total.topic.k.misc',
            21: 'total.topic.k.other',
            22: 'top.topic.k',
            23: 'dist.k.topic_model',
            24: 'total.emb.similar.loc',
            25: 'total.emb.similar.org',
            26: 'total.emb.similar.per',
            27: 'total.emb.similar.misc',
            28: 'total.emb.similar.other',
            29: 'stats.topic.top.k.sum.loc',
            30: 'stats.topic.top.k.sum.org',
            31: 'stats.topic.top.k.sum.per',
            32: 'stats.topic.top.k.sum.misc',
            33: 'stats.topic.top.k.sum.other',
            34: 'stats.topic.top.k.avg.loc',
            35: 'stats.topic.top.k.avg.org',
            36: 'stats.topic.top.k.avg.per',
            37: 'stats.topic.top.k.avg.misc',
            38: 'stats.topic.top.k.avg.other',
            39: 'stats.topic.top.k.max.loc',
            40: 'stats.topic.top.k.max.org',
            41: 'stats.topic.top.k.max.per',
            42: 'stats.topic.top.k.max.misc',
            43: 'stats.topic.top.k.max.other',
            44: 'stats.topic.top.k.min.loc',
            45: 'stats.topic.top.k.min.org',
            46: 'stats.topic.top.k.min.per',
            47: 'stats.topic.top.k.min.misc',
            48: 'stats.topic.top.k.min.other',
            49: 'stats.topic.sum.loc',
            50: 'stats.topic.sum.org',
            51: 'stats.topic.sum.per',
            52: 'stats.topic.sum.misc',
            53: 'stats.topic.sum.other',
            54: 'stats.topic.avg.loc',
            55: 'stats.topic.avg.org',
            56: 'stats.topic.avg.per',
            57: 'stats.topic.avg.misc',
            58: 'stats.topic.avg.other',
            59: 'stats.topic.max.loc',
            60: 'stats.topic.max.org',
            61: 'stats.topic.max.per',
            62: 'stats.topic.max.misc',
            63: 'stats.topic.max.other',
            64: 'stats.topic.min.loc',
            65: 'stats.topic.min.org',
            66: 'stats.topic.min.per',
            67: 'stats.topic.min.misc',
            68: 'stats.topic.min.other'
        }

        reversed_features = dict([(value, key) for key, value in features.items()])

        return features, reversed_features

    @staticmethod
    def get_lexical() -> dict:
        features =  {
            0: 'word.lower',
            1: 'word.lemma',
            2: 'word.stem',
            3: 'word.len.1',
            4: 'word.has.special',
            5: 'word[0].isupper',
            6: 'word.isupper',
            7: 'word.istitle',
            8: 'word.isdigit',
            9: 'word.stop',
            10: 'word.len.issmall',
            11: 'word.has.minus',
            12: 'word.shape',
            13: 'brown_320.1',
            14: 'brown_320.2',
            15: 'brown_320.3',
            16: 'brown_320.4',
            17: 'brown_320.5',
            18: 'brown_640.1',
            19: 'brown_640.2',
            20: 'brown_640.3',
            21: 'brown_640.4',
            22: 'brown_640.5',
            23: 'brown_1000.1',
            24: 'brown_1000.2',
            25: 'brown_1000.3',
            26: 'brown_1000.4',
            27: 'brown_1000.5'
        }

        reversed_features = dict([(value, key) for key, value in features.items()])

        return features, reversed_features


class HorusWordFeatures(object):
    def __init__(self,
                 alias: str,
                 acronym: str,
                 dictionary_size: int,
                 values: [] = None,
                 db_id: int = None):

        self.db_id = db_id
        self.alias = alias
        self.acronym = acronym
        self.dictionary_size = dictionary_size

        if values is None:
            self.values = [0] * dictionary_size
        else:
            self.values = values

        assert self.dictionary_size == len(self.values)

        #def get_feature_idx_by_alias(feature_alias: str):
        #    for idx, value in self.features_dictionary.items():
        #        if feature_alias == value:
        #            return idx
        #    raise Exception('feature not found! check the feature dict', feature_alias)


class HorusFeaturesSet(object):
    def __init__(self,
                 lexical: HorusWordFeatures = None,
                 text: HorusWordFeatures = None,
                 image: HorusWordFeatures = None):

        if lexical is None:
            d, dr = WordFeaturesInterface.get_lexical()
            self.lexical = HorusWordFeatures(alias='Lexical',
                                             acronym='LX',
                                             dictionary_size=len(d))
        else:
            self.lexical = lexical

        if text is None:
            d, dr = WordFeaturesInterface.get_textual()
            self.text = HorusWordFeatures(alias='Text',
                                          acronym='TX',
                                          dictionary_size=len(d))
        else:
            self.text = text

        if image is None:
            d, dr = WordFeaturesInterface.get_visual()
            self.image = HorusWordFeatures(alias='Image',
                                           acronym='CV',
                                           dictionary_size=len(d))
        else:
            self.image = image


class HorusToken(object):
    def __init__(self,
                 text: str,
                 text_original: str,
                 begin_index: int,
                 end_index: int,
                 pos: str = None,
                 pos_probability: float = 1.0,
                 ner: str = None,
                 ner_uri: str = None,
                 language: str = None,
                 is_compound: bool = False,
                 features: HorusFeaturesSet = None):
        '''
        :param text:
        :param begin_index:
        :param end_index:
        :param pos:
        :param pos_probability:
        :param ner:
        :param ner_uri:
        :param language:
        :param is_compound:
        :param features:
        '''
        self.text = text
        self.text_original = text_original
        self.begin_index = begin_index
        self.end_index = end_index
        self.label_pos = pos
        self.label_pos_prob = pos_probability
        self.label_ner = None
        self.label_ner_uri = ner_uri
        self.label_ner_gold = ner
        self.language = language
        self.is_compound = is_compound
        if features is None:
            self.features = HorusFeaturesSet()

    def set_predicted_ner(self, y):
        self.label_ner = y

    def _check_feature_set(self):
        if self.features is None:
            self.features = HorusFeaturesSet()

    def set_feature_lexical(self, feat: HorusWordFeatures) -> bool:
        self._check_feature_set()
        self.features.lexical = feat

    def set_feature_tx(self, feat: HorusWordFeatures) -> bool:
        self._check_feature_set()
        self.features.text = feat

    def set_feature_cv(self, feat: HorusWordFeatures) -> bool:
        self._check_feature_set()
        self.features.image = feat


class HorusSentence(object):
    def __init__(self, index: int = None, text: str = None, text_no_space: str = None, tokens: [] = []):
        self.index = index
        self.text = text
        self.text_no_space = text_no_space
        self.tokens = tokens

    def get_token_index_by_position(self, token_tokenizer: str, begin_index: int, end_index: int, last_token_idx_checkpoint: int = 0) -> int:
        idxs = []
        for i in range(last_token_idx_checkpoint, len(self.tokens)):
            # TOKENIZER groups 1+ CONLL tokens.
            if self.tokens[i].begin_index >= begin_index and self.tokens[i].end_index <= end_index:
                idxs.extend([i])
                if self.tokens[i].end_index > end_index:
                    return idxs
            # TOKENIZER SPLITS 1 CONLL token into 1+ tokens.
            elif self.tokens[i].begin_index <= begin_index and self.tokens[i].end_index >= end_index and \
                    token_tokenizer in self.tokens[i].text:
                return [i]
        return idxs

    def add_token(self, token: HorusToken):
        self.tokens.append(token)

    def get_sentence_no_space(self) -> str:
        if self.text_no_space is not None and len(self.text_no_space) > 0:
            return self.text_no_space
        self.text_no_space = ''
        for t in self.tokens:
            #unescaped = html.unescape(t.text)
            self.text_no_space += t.text
        self.text_no_space = str.strip(self.text_no_space)
        return self.text_no_space

    def get_sentence(self) -> str:
        if self.text is not None and len(self.text) > 0:
            return self.text
        self.text = ''
        for t in self.tokens:
            # unescaped = html.unescape(t.text)
            #if t.text in PUNCTUATION_AND_OTHERS or t.text[0] == '\'':
            if t.text[0] == '\'':
                #self.text = str.strip(self.text) + unescaped + ' '
                self.text = str.strip(self.text) + t.text + ' '
            else:
                #self.text += unescaped + ' '
                self.text += t.text + ' '

        self.text = str.strip(self.text)
        return self.text

    def set_index(self, index: int):
        self.index = index


class Horus(object):
    def __init__(self,
                 dataset: str,
                 language: str,
                 sentences: [] = [],
                 status: str = str(PRE_PROCESSING_STATUS["TOKENIZATION_POS"]),
                 train_on_features_lx: [] = [],
                 train_on_features_tx: [] = [],
                 train_on_features_cv: [] = []):
        self.dataset = dataset
        self.language = language
        self.sentences = sentences
        self.processing_status = status
        self.train_on_features_lx = train_on_features_lx
        self.train_on_features_tx = train_on_features_tx
        self.train_on_features_cv = train_on_features_cv

    def update_status(self, status: int = PRE_PROCESSING_STATUS):
        # concatenate the status so that we known which step went through without needing to perform a specific order
        status = str(status)
        self.processing_status += status

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__)

    def add_sentence(self, sentence: HorusSentence):
        self.sentences.append(sentence)

    def export_to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__)


class HorusDataLoader:
    @staticmethod
    def save_metadata_to_file(horus: Horus, file: str, simple_json: bool = True):
        if simple_json:
            with open(file, 'w') as file:
                file.write(horus.to_json())
        else:
            _json = jsonpickle.encode(horus, unpicklable=True)
            with open(file, 'w') as json_file:
                json.dump(_json, json_file)

    @staticmethod
    def load_metadata_from_file(file: str) -> Horus:
        with open(file) as json_file:
            data = json.load(json_file)
        return jsonpickle.decode(data)


if __name__ == '__main__':

    t1 = HorusToken(text='ewrwer', text_original='adas', begin_index=0, end_index=1)
    t2 = HorusToken(text='thewris2', text_original='adas2', begin_index=3, end_index=6)
    t3 = HorusToken(text='thirwes3', text_original='adas3', begin_index=6, end_index=7)


    tokens = [t1, t2, t3]

    sentences = []
    s = HorusSentence(1, 'this is a sentence', tokens)
    sentences.append(s)

    x = Horus('ritter', 'en', sentences)
    print(x.export_to_json())

    HorusDataLoader.save_metadata_to_file(horus=x,
                                          file='horus.json',
                                          simple_json=False)

    y = HorusDataLoader.load_metadata_from_file(file='horus.json')
    print(y.export_to_json())


from functools import lru_cache
from config import HorusConfig
from src import definitions
from src.definitions import encoder_le1_name, PRE_PROCESSING_STATUS
from src.horus_meta import Horus, HorusDataLoader, WordFeaturesInterface
from nltk import LancasterStemmer, re, WordNetLemmatizer
from nltk import WordNetLemmatizer, SnowballStemmer
from sklearn.externals import joblib
from nltk.corpus import stopwords


def _append_word_lemma_stem(w, l, s):
    t = []
    try:
        t.append(enc_word.transform(str(w)))
    except:
        config.logger.warn('enc_word.transform error')
        t.append(0)

    try:
        t.append(enc_lemma.transform(l.decode('utf-8')))
    except:
        config.logger.warn('enc_lemma.transform error')
        t.append(0)

    try:
        t.append(enc_stem.transform(s.decode('utf-8')))
    except:
        config.logger.warn('enc_stem.transform error')
        t.append(0)

    return t


def _shape(word):
    word_shape = 0  # 'other'
    if re.match('[0-9]+(\.[0-9]*)?|[0-9]*\.[0-9]+$', word):
        word_shape = 1  # 'number'
    elif re.match('\W+$', word):
        word_shape = 2  # 'punct'
    elif re.match('[A-Z][a-z]+$', word):
        word_shape = 3  # 'capitalized'
    elif re.match('[A-Z]+$', word):
        word_shape = 4  # 'uppercase'
    elif re.match('[a-z]+$', word):
        word_shape = 5  # 'lowercase'
    elif re.match('[A-Z][a-z]+[A-Z][a-z]+[A-Za-z]*$', word):
        word_shape = 6  # 'camelcase'
    elif re.match('[A-Za-z]+$', word):
        word_shape = 7  # 'mixedcase'
    elif re.match('__.+__$', word):
        word_shape = 8  # 'wildcard'
    elif re.match('[A-Za-z0-9]+\.$', word):
        word_shape = 9  # 'ending-dot'
    elif re.match('[A-Za-z0-9]+\.[A-Za-z0-9\.]+\.$', word):
        word_shape = 10  # 'abbreviation'
    elif re.match('[A-Za-z0-9]+\-[A-Za-z0-9\-]+.*$', word):
        word_shape = 11  # 'contains-hyphen'

    return word_shape


def _extract_lexical(horus: Horus) -> bool:

    try:
        lx_dict, lx_dict_reversed = WordFeaturesInterface.get_lexical()
        tot_slide_brown_cluster = 5
        for sentence in horus.sentences:
            for token in sentence.tokens:
                brown_1000_path = '{:<016}'.format(dict_brown_c1000.get(token.text, '0000000000000000'))
                brown_640_path = '{:<016}'.format(dict_brown_c640.get(token.text, '0000000000000000'))
                brown_320_path = '{:<016}'.format(dict_brown_c320.get(token.text, '0000000000000000'))

                for i in range(0, tot_slide_brown_cluster - 1):
                    token.features.lexical.values[lx_dict_reversed.get('brown_1000.' + str(i + 1))] = brown_1000_path[
                                                                                                      :i + 1]
                    token.features.lexical.values[lx_dict_reversed.get('brown_640.' + str(i + 1))] = brown_640_path[
                                                                                                     :i + 1]
                    token.features.lexical.values[lx_dict_reversed.get('brown_320.' + str(i + 1))] = brown_320_path[
                                                                                                     :i + 1]

                token.features.lexical.values[lx_dict_reversed.get('word.lower')] = token.text.lower()

                lemma = ''
                try:
                    lemma = lemmatize(token.text.lower())
                except:
                    pass

                stem = ''
                try:
                    stem = stemo(token.text.lower())
                except:
                    pass

                token.features.lexical.values[lx_dict_reversed.get('word.lemma')] = lemma
                token.features.lexical.values[lx_dict_reversed.get('word.stem')] = stem
                token.features.lexical.values[lx_dict_reversed.get('word.len.1')] = int(len(token.text) == 1)
                token.features.lexical.values[lx_dict_reversed.get('word.has.special')] = int(
                    len(re.findall('(http://\S+|\S*[^\w\s]\S*)', token.text)) > 0)
                token.features.lexical.values[lx_dict_reversed.get('word[0].isupper')] = int(token.text[0].isupper())
                token.features.lexical.values[lx_dict_reversed.get('word.isupper')] = int(token.text.isupper())
                token.features.lexical.values[lx_dict_reversed.get('word.istitle')] = int(token.text.istitle())
                token.features.lexical.values[lx_dict_reversed.get('word.isdigit')] = int(token.text.isdigit())
                token.features.lexical.values[lx_dict_reversed.get('word.len.issmall')] = int(len(token.text) <= 2)
                token.features.lexical.values[lx_dict_reversed.get('word.has.minus')] = int('-' in token.text)
                token.features.lexical.values[lx_dict_reversed.get('word.stop')] = int(token.text in stop)
                token.features.lexical.values[lx_dict_reversed.get('word.shape')] = _shape(token.text)

        return True

    except Exception as e:
        raise e


def _extract_text(horus: Horus) -> bool:
    '''
    Please see src/training/ for more information
    :param horus:
    :return:
    '''
    try:
        tx_dict, tx_dict_reversed = WordFeaturesInterface.get_textual()
        for sentence in horus.sentences:
            for token in sentence.tokens:
                brown_1000_path = '{:<016}'.format(dict_brown_c1000.get(token.text, '0000000000000000'))
                brown_640_path = '{:<016}'.format(dict_brown_c640.get(token.text, '0000000000000000'))
                brown_320_path = '{:<016}'.format(dict_brown_c320.get(token.text, '0000000000000000'))

                for i in range(0, tot_slide_brown_cluster - 1):
                    token.features.lexical.values[lx_dict_reversed.get('brown_1000.' + str(i + 1))] = brown_1000_path[
                                                                                                      :i + 1]
                    token.features.lexical.values[lx_dict_reversed.get('brown_640.' + str(i + 1))] = brown_640_path[
                                                                                                     :i + 1]
                    token.features.lexical.values[lx_dict_reversed.get('brown_320.' + str(i + 1))] = brown_320_path[
                                                                                                     :i + 1]

                token.features.lexical.values[lx_dict_reversed.get('word.lower')] = token.text.lower()

                lemma = ''
                try:
                    lemma = lemmatize(token.text.lower())
                except:
                    pass

                stem = ''
                try:
                    stem = stemo(token.text.lower())
                except:
                    pass

                token.features.lexical.values[lx_dict_reversed.get('word.lemma')] = lemma
                token.features.lexical.values[lx_dict_reversed.get('word.stem')] = stem
                token.features.lexical.values[lx_dict_reversed.get('word.len.1')] = int(len(token.text) == 1)
                token.features.lexical.values[lx_dict_reversed.get('word.has.special')] = int(
                    len(re.findall('(http://\S+|\S*[^\w\s]\S*)', token.text)) > 0)
                token.features.lexical.values[lx_dict_reversed.get('word[0].isupper')] = int(token.text[0].isupper())
                token.features.lexical.values[lx_dict_reversed.get('word.isupper')] = int(token.text.isupper())
                token.features.lexical.values[lx_dict_reversed.get('word.istitle')] = int(token.text.istitle())
                token.features.lexical.values[lx_dict_reversed.get('word.isdigit')] = int(token.text.isdigit())
                token.features.lexical.values[lx_dict_reversed.get('word.len.issmall')] = int(len(token.text) <= 2)
                token.features.lexical.values[lx_dict_reversed.get('word.has.minus')] = int('-' in token.text)
                token.features.lexical.values[lx_dict_reversed.get('word.stop')] = int(token.text in stop)
                token.features.lexical.values[lx_dict_reversed.get('word.shape')] = _shape(token.text)

        return True

    except Exception as e:
        raise e


def _extract_visual(horus: Horus) -> bool:
    '''
    Please see src/training/ for more information
    :param horus:
    :return:
    '''
    return True


def extract_features(horus: Horus, lexical: bool = False, text: bool = False, image: bool = False) -> bool:
    try:

        if lexical:
            _extract_lexical(horus)

        if text:
            _extract_text(horus)

        # TODO: implement
        if image:
            _extract_visual(horus)

        return True

    except Exception as e:
        config.logger.exception(str(e))
        return False


if __name__ == '__main__':

    config = HorusConfig()

    # define the feature sets you want to extract
    EXTRACT_LEXICAL = True
    EXTRACT_TEXT = False
    EXTRACT_IMAGE = False

    config.logger.info('loading lemmatizers')
    stemmer = SnowballStemmer('english')
    stop = set(stopwords.words('english'))
    wnl = WordNetLemmatizer()

    lemmatize = lru_cache(maxsize=50000)(wnl.lemmatize)
    stemo = lru_cache(maxsize=50000)(stemmer.stem)

    config.logger.info('loading encoders')
    enc_le1 = joblib.load(config.dir_encoders + definitions.encoder_le1_name)
    enc_le2 = joblib.load(config.dir_encoders + definitions.encoder_le2_name)
    enc_word = joblib.load(config.dir_encoders + definitions.encoder_int_words_name)
    enc_lemma = joblib.load(config.dir_encoders + definitions.encoder_int_lemma_name)
    enc_stem = joblib.load(config.dir_encoders + definitions.encoder_int_stem_name)
    le = joblib.load(config.dir_encoders + encoder_le1_name)

    config.logger.info('loading brown corpus')
    dict_brown_c1000 = joblib.load(config.dir_clusters + 'gha.500M-c1000-p1.paths_dict.pkl')
    dict_brown_c640 = joblib.load(config.dir_clusters + 'gha.64M-c640-p1.paths_dict.pkl')
    dict_brown_c320 = joblib.load(config.dir_clusters + 'gha.64M-c320-p1.paths_dict.pkl')

    config.logger.info('done')

    # initialize the horus metadata file for each dataset
    for ds in definitions.NER_DATASETS:
        try:
            conll_file = ds[1] + ds[2]
            assert '.horusx' in conll_file
            horus_file_stage2 = conll_file.replace('.horusx', '.horus2.json')

            config.logger.info('loading horus file: ' + horus_file_stage2)
            horus = HorusDataLoader.load_metadata_from_file(file=horus_file_stage2)

            config.logger.info(f'feature extraction: '
                               f'lexical: {EXTRACT_LEXICAL}, text: {EXTRACT_TEXT}, image: {EXTRACT_IMAGE}')
            ok = extract_features(horus, lexical=EXTRACT_LEXICAL, text=EXTRACT_TEXT, image=EXTRACT_IMAGE)
            if not ok:
                config.logger.warn('feature extraction: something went wrong...')

            config.logger.info('updating status')
            horus.update_status(PRE_PROCESSING_STATUS["FEATURE_LEXICAL"])

            config.logger.info('done! saving files')

            horus_file_stage3_simple_json = conll_file.replace('.horusx', '.horus3.simple.json')
            horus_file_stage3 = conll_file.replace('.horusx', '.horus3.json')

            # TODO: for now I am saving in a different json file just to compare and check things are fine.
            # later just update the status of the horus file (definitions.PRE_PROCESSING_STATUS)
            HorusDataLoader.save_metadata_to_file(horus=horus, file=horus_file_stage3_simple_json, simple_json=True)
            HorusDataLoader.save_metadata_to_file(horus=horus, file=horus_file_stage3, simple_json=False)

            config.logger.info('all good!')

        except Exception as e:
            config.logger.error(str(e))
            continue

import csv
import pickle

import nltk
import numpy

from src.config import HorusConfig
from django.core.exceptions import ValidationError
from django.core.validators import URLValidator
from time import gmtime, strftime
import requests
import pandas as pd
import re
import os

from src.util import definitions
from src.util.definitions import INDEX_ID_TERM_TXT, INDEX_ID_TERM_IMG
from src.util.definitions_sql import SQL_ALL_TERM_SEARCH_SEL
from src.util.nlp_tools import NLPTools
from src.util.search_engines import query_flickr, query_wikipedia, query_bing
from src.util.sqlite_helper import HorusDB, SQLiteHelper

config = HorusConfig()
tools = NLPTools(config)

def tokenize_and_pos(sentence, annotator_id=3):
    # NLTK
    if annotator_id == 1:
        return tools.tokenize_and_pos_nltk(sentence)
    # Stanford
    elif annotator_id == 2:
        return tools.tokenize_and_pos_stanford(sentence)
    # TwitterNLP
    elif annotator_id == 3:
        if type(sentence) is not list:
            return tools.tokenize_and_pos_twitter(sentence)
        return tools.tokenize_and_pos_twitter_list(sentence)

def process_ds_conll_format(dspath, token_index=0, ner_index=1, separator='\t'):
    '''
    return a set of sentences
    :param dspath: path to Ritter dataset
    :return: sentence contains any entity?, sentence, words, NER tags
    '''
    try:
        import codecs
        sentences = []
        tokens = []
        tags_ner_y = []
        s = ''
        has3NER = -1
        tot_sentences = 1
        config.logger.info('processing sentences...')
        with open(dspath) as f:
            docstart = False
            for line in f:
                if line.strip() != '':
                    if separator == '': separator = None
                    splited = line.split(separator)
                    if len(splited) < 2: #solving a specific bug
                        token='.'
                        ner='O'
                        config.logger.error('attention!!!! this can not happen more than once or twice...')
                    else:
                        token = splited[token_index]
                        ner = splited[ner_index].replace('\r', '').replace('\n', '')
                if line.strip() == '':
                    if docstart is False:
                        if len(tokens) != 0:
                            config.logger.debug('processing sentence %s' % str(tot_sentences))
                            if tot_sentences==1008:
                                a=1
                            sentences.append(process_sentence(has3NER, s, tokens, tags_ner_y))
                            tokens = []
                            tags_ner_y = []
                            s = ''
                            has3NER = -1
                            tot_sentences += 1
                            if tot_sentences % 100 == 0:
                                config.logger.info('caching sentence: %s' % str(tot_sentences))
                    else:
                        docstart = False
                else:
                    if token != '-DOCSTART-':
                        s += token + ' '
                        tokens.append(token)
                        tags_ner_y.append(ner)
                        if ner in definitions.NER_TAGS:
                            has3NER = 1
                    else:
                        docstart = True

        config.logger.info('%s sentences processed successfully' % str(len(sentences)))
        return sentences
    except Exception as error:
        config.logger.error('caught this error: ' + repr(error))

def get_ner_mapping_simple(y, x, ix, starty):
    try:
        index = -1
        for k in range(starty, len(y)):
            base = y[k]
            for i in range(ix, len(x)):
                term = x[i]
                if config.models_pos_tag_lib == 1:  # nltk
                    term = x[i].replace('``', u'"')

                swap = ''
                if config.models_pos_tag_lib != 3:
                    if term == "''": swap = '"'
                    if term == '"': swap = "''"
                # tweetNLP
                # if u'&amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;lt' == x[i]:
                #    term = term.replace(u'&amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;lt', u'&lt;')
                # elif u'&' in x[i]:
                #    term = term.replace(u'&', u'&amp;')
                # elif u'<' in x[i]:
                #    term = term.replace(u'<', u'&lt;')
                # elif u'>' in x[i]:
                #    term = term.replace(u'>', u'&gt;')

                if config.models_pos_tag_lib == 3:
                    base = re.sub("&amp;", "&", base)
                    base = re.sub("&quot;", '"', base)
                    base = re.sub("&apos;", "'", base)
                    base = re.sub("&gt;", ">", base)
                    base = re.sub("&lt;", "<", base)
                    term = re.sub("&amp;", "&", term)
                    term = re.sub("&quot;", '"', term)
                    term = re.sub("&apos;", "'", term)
                    term = re.sub("&apos", "'", term)  # trick
                    term = re.sub("&gt;", ">", term)
                    term = re.sub("&lt;", "<", term)

                if term in base or (swap in base if swap != '' else False):
                    index = k
                    if i == ix:
                        return index
        raise Exception

    except Exception as error:
        config.logger.error(repr(error))
        raise

def __get_compounds(tokens):
    compounds = []
    pattern = """
                                NP:
                                   {<JJ>*<NN|NNS|NNP|NNPS><CC>*<NN|NNS|NNP|NNPS>+}
                                   {<NN|NNS|NNP|NNPS><IN>*<NN|NNS|NNP|NNPS>+}
                                   {<JJ>*<NN|NNS|NNP|NNPS>+}
                                   {<NN|NNP|NNS|NNPS>+}
                                """
    cp = nltk.RegexpParser(pattern)
    toparse = []
    for token in tokens:
        toparse.append(tuple([token[0], token[1]]))
    t = cp.parse(toparse)

    i_word = 0
    for item in t:
        if type(item) is nltk.Tree:
            i_word += len(item)
            if len(item) > 1:  # that's a compound
                compound = ''
                for tk in item:
                    compound += tk[0] + ' '

                compounds.append([i_word - len(item) + 1, compound[:len(compound) - 1], len(item)])
        else:
            i_word += 1
    return compounds

def process_sentence(hasNER, s, tokens_gold_standard=[], ner_gold_standard=[]):

    try:
        _tokens_twe, _pos_twe, _pos_uni_twe = tokenize_and_pos(s)
        _pos_twe = numpy.array(_pos_twe)
        _pos_uni_twe = numpy.array(_pos_uni_twe)
        _same_tok_tweet = (len(_tokens_twe) == len(tokens_gold_standard))
        comp_twe = __get_compounds(_pos_twe)

        # saving to database (pos_uni_sta not implemented yet)
        sent = [hasNER,
                [s,
                 0, # 1 if _same_tok_nltk else 0
                 0, # 1 if _same_tok_stanf else 0,
                 1 if _same_tok_tweet else 0
                 ],
                [tokens_gold_standard,
                 [], #_tokens_nltk,
                 [], # _tokens_st,
                 _tokens_twe
                 ],
                [ner_gold_standard,
                 [], #nernltktags,
                 [], # nerstantags[:, 1].tolist(),
                 []
                 ],
                [[],
                 [], # _pos_nltk[:, 1].tolist(),
                 [], #  _pos_st[:, 1].tolist(),
                 _pos_twe[:, 1].tolist() if len(_pos_twe)>0 else []
                 ],
                [[],
                 [], #_pos_uni_nltk[:, 1].tolist(),
                 [],
                 _pos_uni_twe[:, 1].tolist() if len(_pos_uni_twe)>0 else []],
                [[],
                 [], #comp_nltk,
                 [], # comp_st,
                 comp_twe]
                ]
        return sent
    except Exception as e:
        config.logger.error(repr(e))
        raise



def populate_matrix_new_columns():
    temp = []
    temp.extend([0] * (int(definitions.HORUS_TOT_FEATURES)-8))
    # do NOT append the last column here (y)

    temp[18] = definitions.PLOMNone_label2index["O"]
    temp[26] = definitions.PLOMNone_label2index["O"]
    temp[26] = definitions.PLOMNone_label2index["O"]
    temp[38] = definitions.PLOMNone_label2index["O"]
    temp[39] = definitions.PLOMNone_label2index["O"]
    temp[40] = definitions.PLOMNone_label2index["O"]
    temp[41] = definitions.PLOMNone_label2index["O"]

    return temp

def sentence_to_horus_matrix(sentences):
    '''
    converts the list to horus_matrix
    :param sentences
    :return: horus_matrix
    '''
    config.logger.info('starting conversion to horus_matrix based on system parameters')
    converted = []
    sent_index = 0
    try:
        for sent in sentences:
            sent_index += 1
            ipositionstartterm = 0
            for c in range(len(sent[6][config.models_pos_tag_lib])):
                word_index_ref = sent[6][config.models_pos_tag_lib][c][0]
                compound = sent[6][config.models_pos_tag_lib][c][1]
                compound_size = sent[6][config.models_pos_tag_lib][c][2]
                temp = [0, sent_index, word_index_ref, compound, '', '', definitions.PLOMNone_index2label[4], 1, compound_size]
                temp.extend(populate_matrix_new_columns())
                temp[definitions.INDEX_TARGET_NER] = "O"
                converted.append(temp)
            word_index = 0
            starty = 0
            for i in range(len(sent[2][config.models_pos_tag_lib])):
                term = sent[2][config.models_pos_tag_lib][i]
                if len(sent[2][0]) > 0:
                    ind_ner_real = get_ner_mapping_simple(sent[2][0], sent[2][config.models_pos_tag_lib], i, starty)
                    starty = ind_ner_real
                    # ind_ner = get_ner_mapping_slice(sent[2][0], sent[2][config.models_pos_tag_lib], i)
                    # ind_ner = get_ner_mapping2(sent[2][0], sent[2][config.models_pos_tag_lib], term, i)
                    has_NER = 1 if sent[3][0][ind_ner_real] in definitions.NER_TAGS else 0
                else:
                    has_NER = -1
                tag_ner = sent[3][config.models_pos_tag_lib][i] if len(
                    sent[3][config.models_pos_tag_lib]) > 0 else ''
                tag_pos = sent[4][config.models_pos_tag_lib][i] if len(
                    sent[4][config.models_pos_tag_lib]) > 0 else ''
                tag_pos_uni = sent[5][config.models_pos_tag_lib][i] if len(
                    sent[5][config.models_pos_tag_lib]) > 0 else ''
                word_index += 1
                # we do not know if they have the same alignment, so test it to get the correct tag
                if len(sent[3][0]) > 0:
                    tag_ner_y = sent[3][0][ind_ner_real]
                    if tag_ner_y in definitions.NER_TAGS_LOC:
                        tag_ner_y = definitions.PLOMNone_label2index["LOC"]
                    elif tag_ner_y in definitions.NER_TAGS_ORG:
                        tag_ner_y = definitions.PLOMNone_label2index["ORG"]
                    elif tag_ner_y in definitions.NER_TAGS_PER:
                        tag_ner_y = definitions.PLOMNone_label2index["PER"]
                    elif tag_ner_y in definitions.NER_TAGS_MISC:
                        tag_ner_y = definitions.PLOMNone_label2index["MISC"]
                    else:
                        tag_ner_y = definitions.PLOMNone_label2index["O"]
                else:
                    tag_ner_y = definitions.PLOMNone_label2index["O"]

                if tag_ner in definitions.NER_TAGS_LOC:
                    tag_ner = definitions.PLOMNone_label2index["LOC"]
                elif tag_ner in definitions.NER_TAGS_ORG:
                    tag_ner = definitions.PLOMNone_label2index["ORG"]
                elif tag_ner in definitions.NER_TAGS_PER:
                    tag_ner = definitions.PLOMNone_label2index["PER"]
                elif tag_ner in definitions.NER_TAGS_MISC:
                    tag_ner = definitions.PLOMNone_label2index["MISC"]
                else:
                    tag_ner = definitions.PLOMNone_label2index["O"]

                temp = [has_NER, sent_index, word_index, term, tag_pos_uni, tag_pos, tag_ner, 0, 0]  # 0-8
                temp.extend(populate_matrix_new_columns())
                temp[definitions.INDEX_TARGET_NER] = tag_ner_y
                ## that is a hack to integrate to GERBIL
                # if ipositionstartterm >= len(sent[1][0]):
                #    ipositionstartterm-=1
                # if sent[1][0][ipositionstartterm] == term[0]:
                #    if sent[1][0][ipositionstartterm:ipositionstartterm+len(term)] != term:
                #        raise Exception("GERBIL integration: error 1!")
                # else:
                #    ipositionstartterm-=1
                #    if sent[1][0][ipositionstartterm] == term[0]:
                #        if sent[1][0][ipositionstartterm:ipositionstartterm+len(term)] != term:
                #            raise Exception("GERBIL integration: error 2!")
                #    else:
                #        raise Exception("GERBIL integration: error 3!")

                #temp[27] = ipositionstartterm
                converted.append(temp)
                ipositionstartterm += (len(term) + 1)

    except:
        raise

    return converted

def __get_horus_matrix_structure(sent_tokenize_list):

    df = pd.DataFrame(sent_tokenize_list)

    config.logger.info('%s sentence(s) cached' % str(len(sent_tokenize_list)))
    tot_sentences_with_entity = len(df.loc[df[0] == 1])
    tot_others = len(df.loc[df[0] == -1])
    config.logger.info('%s sentence(s) with entity' % tot_sentences_with_entity)
    config.logger.info('%s sentence(s) without entity' % tot_others)
    horus_matrix = sentence_to_horus_matrix(sent_tokenize_list)

    hm = pd.DataFrame(horus_matrix)
    config.logger.info('basic POS statistics')
    a = len(hm)
    # all excluding compounds
    a2 = len(hm[(hm[definitions.INDEX_IS_COMPOUND] == 0)])
    # all PLO entities (not compound)
    plo = hm[(hm[definitions.INDEX_IS_COMPOUND] == 0) & (hm[definitions.INDEX_IS_ENTITY] == 1)]
    # all PLO entities (not compound)
    not_plo = hm[(hm[7] == 0) & (hm[0] == 0)]
    from collections import Counter
    config.logger.info('[POS tags counter for ALL tokens]')
    config.logger.info(Counter(hm[definitions.INDEX_POS]))
    config.logger.info('[POS tags counter for PLO tokens]')
    config.logger.info(Counter(plo[definitions.INDEX_POS]))
    pos_ok_plo = plo[(plo[definitions.INDEX_POS].isin(definitions.POS_NOUN_TAGS))]
    pos_not_ok_plo = plo[(~plo[definitions.INDEX_POS].isin(definitions.POS_NOUN_TAGS))]
    pos_noun_but_not_entity = not_plo[(not_plo[definitions.INDEX_POS].isin(definitions.POS_NOUN_TAGS))]
    config.logger.info('[basic statistics]')
    config.logger.info('-> ALL terms: %s ' % a)
    config.logger.info('-> ALL tokens (no compounds): %s (%.2f)' % (a2, (a2 / float(a))))
    config.logger.info('-> ALL NNs (no compounds nor entities): %s ' % len(pos_noun_but_not_entity))
    config.logger.info('[test dataset statistics]')
    config.logger.info('-> PLO entities (no compounds): %s (%.2f)' % (len(plo), len(plo) / float(a2)))
    config.logger.info('-> PLO entities correctly classified as NN (POS says is NOUN): %s (%.2f)' %
                      (len(pos_ok_plo), len(pos_ok_plo) / float(len(plo)) if len(plo) != 0 else 0))
    config.logger.info('-> PLO entities misclassified (POS says is NOT NOUN): %s (%.2f)' %
                      (len(pos_not_ok_plo), len(pos_not_ok_plo) / float(len(plo)) if len(plo) != 0 else 0))

    return horus_matrix
        
def cache_images_and_text(conll_file, tokenindex=0, ner_index=1, sep='\t'):
    '''
    given an input file, read the content and caches the results
    :param conll_file: a conll inoutfile
    :param tokenindex: the position of the token in the file
    :return: True or False
    '''
    try:
        if conll_file is None:
            raise Exception("Provide an input file format to be annotated")
        config.logger.info('caching file -> %s' % conll_file)

        file_sentences =  conll_file.replace('.horusx', '.horus1')
        file_horus_matrix = conll_file.replace('.horusx', '.horus2')
        conll_file = conll_file.replace('.horusx', '')

        if os.path.isfile(file_horus_matrix):
            config.logger.info('file %s has been processed successfully!' % (file_horus_matrix))
        else:
            if os.path.isfile(file_sentences):
                config.logger.info('loading ' + file_sentences)
                with open(file_sentences, 'rb') as input:
                    sent_tokenize_list=pickle.load(input)
            else:
                config.logger.info('exporting *.horus1')
                sent_tokenize_list = process_ds_conll_format(conll_file, tokenindex, ner_index, sep)
                with open(file_sentences, 'wb') as output:
                    pickle.dump(sent_tokenize_list, output, pickle.HIGHEST_PROTOCOL)

            if len(sent_tokenize_list) > 0:
                config.logger.info('exporting *.horus2')
                horus_matrix = __get_horus_matrix_structure(sent_tokenize_list)
                __download_and_cache_results(horus_matrix)
                writer = csv.writer(open(file_horus_matrix, 'wb'), dialect="excel", delimiter=sep, skipinitialspace=True)
                writer.writerow(definitions.HORUS_MATRIX_HEADER)
                writer.writerows(horus_matrix)
                config.logger.info('process finished! horus_matrix exported: ' + file_horus_matrix)
            else:
                config.logger.warn('well, nothing to do today...')

    except Exception as e:
        config.logger.error(repr(e))

def __download_image_local(image_url, image_type, thumbs_url, thumbs_type, term_id, id_ner_type, seq):
    val = URLValidator()
    auxtype = None
    try:
        val(thumbs_url)
        try:
            img_data = requests.get(thumbs_url).content
            with open('%s%s_%s_%s.%s' % (config.cache_img_folder, term_id, id_ner_type, seq, thumbs_type),
                      'wb') as handler:
                handler.write(img_data)
                auxtype = thumbs_type
        except Exception as error:
            config.logger.error(repr(error))
    except ValidationError:
        config.logger.error('No thumbs img here...')
        try:
            img_data = requests.get(image_url).content
            with open('%s%s_%s_%s.%s' % (config.cache_img_folder, term_id, id_ner_type, seq, image_type),
                      'wb') as handler:
                auxtype = image_type
                handler.write(img_data)
        except Exception as error:
            config.logger.error(repr(error))
    return auxtype

def __download_and_cache_results(horus_matrix):
    try:
        with SQLiteHelper(config.database_db) as sqlcon:
            config.logger.info('caching results...')
            t = HorusDB(sqlcon)
            auxc = 1
            download = False
            try:
                # getting list of cached terms
                values = (config.search_engine_api, config.search_engine_features_text)
                df = pd.read_sql_query(sql=SQL_ALL_TERM_SEARCH_SEL, con=t.conn, params=values)
                df.set_index("id", inplace=True)
                for index in range(len(horus_matrix)):
                    term = horus_matrix[index][definitions.INDEX_TOKEN]
                    term = term.lower()
                    if (horus_matrix[index][definitions.INDEX_POS] in definitions.POS_NOUN_TAGS) or \
                            horus_matrix[index][definitions.INDEX_IS_COMPOUND] == 1:
                        if auxc % 1000 == 0:
                            config.logger.debug(
                                'caching token %s - %s [%s]' % (str(auxc), str(len(horus_matrix)), term))
                        res = df.loc[df['term'] == term]
                        if res is None or len(res) == 0:
                            download = True
                            '''
                            --------------------------------------------------------------------------
                            Downloading resources...
                            --------------------------------------------------------------------------
                            '''
                            config.logger.info('not cached, querying -> [%s]' % term)

                            # Microsoft Bing
                            if int(config.search_engine_api) == 1:
                                metaquery, result_txts, result_imgs = query_bing(term,
                                                                                 key=config.search_engine_key,
                                                                                 top=config.search_engine_tot_resources)
                            # Flickr
                            elif (config.search_engine_api) == 3:
                                metaquery, result_imgs = query_flickr(term)
                                metaquery, result_txts = query_wikipedia(term)

                            '''
                            --------------------------------------------------------------------------
                            Caching Documents (Texts)
                            --------------------------------------------------------------------------
                            '''
                            config.logger.debug('caching %s web sites -> [%s]' % (len(result_txts), term))
                            id_term_search = t.save_term(term, config.search_engine_tot_resources,
                                                         len(result_txts), config.search_engine_api,
                                                         1, config.search_engine_features_text,
                                                         str(strftime("%Y-%m-%d %H:%M:%S", gmtime())), metaquery)
                            horus_matrix[index][definitions.INDEX_ID_TERM_TXT] = id_term_search
                            seq = 0
                            for web_result_txt in result_txts:
                                config.logger.info('caching (web site) -> [%s]' % web_result_txt['displayUrl'])
                                seq += 1
                                t.save_website_data(id_term_search, seq, web_result_txt['id'], web_result_txt['displayUrl'],
                                                    web_result_txt['name'], web_result_txt['snippet'])
                            '''
                            --------------------------------------------------------------------------
                            Caching Documents (Images)
                            --------------------------------------------------------------------------
                            '''
                            config.logger.info('caching %s web images -> [%s]' % (len(result_imgs), term))
                            id_term_img = t.save_term(term, config.search_engine_tot_resources,
                                                      len(result_imgs), config.search_engine_api,
                                                      2, config.search_engine_features_img,
                                                      str(strftime("%Y-%m-%d %H:%M:%S", gmtime())), metaquery)
                            horus_matrix[index][definitions.INDEX_ID_TERM_IMG] = id_term_img
                            seq = 0
                            for web_result_img in result_imgs:
                                config.logger.debug('downloading image [%s]' % (web_result_img['name']))
                                seq += 1
                                auxtype = __download_image_local(web_result_img['contentUrl'],
                                                                    web_result_img['encodingFormat'],
                                                                    web_result_img['thumbnailUrl'],
                                                                    web_result_img['encodingFormat'], id_term_img, 0,
                                                                    seq)
                                config.logger.debug('caching image  ...')
                                t.save_image_data(id_term_img, seq, web_result_img['contentUrl'],
                                                  web_result_img['name'],
                                                  web_result_img['encodingFormat'], web_result_img['height'],
                                                  web_result_img['width'], web_result_img['thumbnailUrl'], str(auxtype))
                            # t.commit()
                            # adding the new item to the cache dataframe
                            config.logger.debug('updating local cache  ...')
                            cols = ['id', 'term', 'id_search_type', 'tot_results_returned']
                            df_txt = pd.DataFrame([[id_term_search, term, 1, len(result_txts)]],
                                                  columns=cols)
                            df_txt.set_index("id", inplace=True)
                            df_img = pd.DataFrame([[id_term_img, term, 2, len(result_imgs)]], columns=cols)
                            df_img.set_index("id", inplace=True)
                            df = pd.concat([df, df_txt, df_img], ignore_index=False, verify_integrity=True)
                            config.logger.debug('OK')

                            # df.conca(df_txt, ignore_index=False, verify_integrity=True)
                            # df=df.append(df_img, ignore_index=False, verify_integrity=True)

                        else:
                            if (len(res) != 2):
                                raise Exception("that should not happen! check db integrity")
                            if ((1) in set(df.loc[(df['term'] == term)]['id_search_type'])):
                                horus_matrix[index][INDEX_ID_TERM_TXT] = \
                                    int(df.loc[(df['term'] == term) & (df['id_search_type'] == 1)].index.values)
                            else:
                                horus_matrix[index][INDEX_ID_TERM_TXT] = -1
                            if ((2) in set(df.loc[(df['term'] == term)]['id_search_type'])):
                                horus_matrix[index][INDEX_ID_TERM_IMG] = \
                                    int(df.loc[(df['term'] == term) & (df['id_search_type'] == 2)].index.values)
                            else:
                                horus_matrix[index][INDEX_ID_TERM_IMG] = -1

                    auxc += 1

                if download:
                    t.commit()
                return horus_matrix

            except Exception as e:
                try:
                    if download:
                        t.commit()
                except:
                    pass
                raise e
    except Exception as e:
        config.logger.error(repr(e))
        raise

if __name__ == "__main__":

    for ds in definitions.NER_DATASETS:
        try:
            cache_images_and_text(ds[1] + ds[2])
            config.logger.info('---')
        except:
            continue

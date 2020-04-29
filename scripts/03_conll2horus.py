# -*- coding: utf-8 -*-
from config import HorusConfig
from src import definitions
import numpy
from src.horus_meta import Horus, HorusSentence, HorusToken, HorusDataLoader
from src.utils.nlp_tools import NLPTools, POS_TAGGER_TWITTERNLP
import nltk
import os
from src.utils.pre_processing import fully_unescape_token


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


def process_sentence(hasNER, s, tokens_gold_standard=[], ner_gold_standard=[], tools=None):
    try:
        _tokens_twe, _pos_twe, _pos_uni_twe = tokenize_and_pos(s, tools)
        _pos_twe = numpy.array(_pos_twe)
        _pos_uni_twe = numpy.array(_pos_uni_twe)
        _same_tok_tweet = (len(_tokens_twe) == len(tokens_gold_standard))
        comp_twe = __get_compounds(_pos_twe)

        # saving to database (pos_uni_sta not implemented yet)
        sent = [hasNER,
                [s,
                 0,  # 1 if _same_tok_nltk else 0
                 0,  # 1 if _same_tok_stanf else 0,
                 1 if _same_tok_tweet else 0
                 ],
                [tokens_gold_standard,
                 [],  # _tokens_nltk,
                 [],  # _tokens_st,
                 _tokens_twe
                 ],
                [ner_gold_standard,
                 [],  # nernltktags,
                 [],  # nerstantags[:, 1].tolist(),
                 []
                 ],
                [[],
                 [],  # _pos_nltk[:, 1].tolist(),
                 [],  # _pos_st[:, 1].tolist(),
                 _pos_twe[:, 1].tolist() if len(_pos_twe) > 0 else []
                 ],
                [[],
                 [],  # _pos_uni_nltk[:, 1].tolist(),
                 [],
                 _pos_uni_twe[:, 1].tolist() if len(_pos_uni_twe) > 0 else []],
                [[],
                 [],  # comp_nltk,
                 [],  # comp_st,
                 comp_twe]
                ]
        return sent
    except Exception as e:
        config.logger.error(repr(e))
        raise


def tokenize_and_pos(sentence: str,
                     annotator_id: int = definitions.POS_TAGGER_TWITTERNLP,
                     tools: NLPTools = None):
    # NLTK
    if annotator_id == definitions.POS_TAGGER_NLTK:
        raise Exception('not currently supported')
        return tools.tokenize_and_pos_nltk(sentence)
    # Stanford
    elif annotator_id == definitions.POS_TAGGER_STANFORD:
        raise Exception('not currently supported')
        return tools.tokenize_and_pos_stanford(sentence)
    # TwitterNLP
    elif annotator_id == definitions.POS_TAGGER_TWITTERNLP:
        if type(sentence) is not list:
            return tools.tokenize_and_pos_twitter(sentence)
        return tools.tokenize_and_pos_twitter_list(sentence)


def add_pos_tags_to_horus_metadata(horus: Horus, tools: NLPTools, tokenizer_id: int = POS_TAGGER_TWITTERNLP) -> Horus:
    try:
        config.logger.info('processing sentences to add POS...')
        total = 0
        for horus_sent in horus.sentences:
            total += 1
            if total == 26:
                print(1)
            sentence = horus_sent.get_sentence()
            _tokens_twe, _pos_twe, _pos_uni_twe, _probs = \
                tokenize_and_pos(sentence=sentence, annotator_id=tokenizer_id, tools=tools)

            # assert len(_tokens_twe) <= len(horus_sent.tokens)
            token_begin_index = 0  # -1
            token_end_index = 0  # -1
            print('')
            print('HORUS Sentence: ', str(total).zfill(4), ' | ', sentence)
            print(' -- POS tokenizer: ', len(_tokens_twe), ' -> ', _tokens_twe)
            print(' -- CoNLL|HORUS: ', len(horus_sent.tokens), ' -> ',
                  [(token.text, token.begin_index, token.end_index) for token in horus_sent.tokens])
            for i in range(0, len(_pos_twe)):
                token_tokenizer = _pos_twe[i][0]
                token_begin_index = token_end_index  # + 1
                token_end_index = token_begin_index + len(token_tokenizer)
                # if token[0] in PUNCTUATION_AND_OTHERS:
                #    token_begin_index -= 1
                #    token_end_index -= 1
                idxs = horus_sent.get_token_index_by_position(token_tokenizer=token_tokenizer,
                                                              begin_index=token_begin_index,
                                                              end_index=token_end_index)
                print('POS token: ', token_tokenizer, str(token_begin_index).zfill(3), str(token_end_index).zfill(3),
                      '| HORUS token index: ', idxs)

                for idx in idxs:
                    horus_sent.tokens[idx].label_pos = _pos_twe[i][1]
                    horus_sent.tokens[idx].label_pos_prob = _probs[i][1]

        return horus

    except Exception as e:
        config.logger.error(str(e))
        raise e


def conll2horus_transform_basic_structure(dspath: str, dataset: str, token_index: int = 0, ner_index: int = 1,
                                          separator: str = '\t', language: str = 'en'):
    try:
        config.logger.info('processing basic transformation...')

        horus = Horus(dataset=dataset, language=language)
        horus_sent = HorusSentence(index=0)

        # last_token_end_index = -1
        last_token_end_index = 0
        tot_sentences = 0

        with open(dspath) as document:
            for line in document:
                if line != '' and line != '\n':
                    split = line.split(separator)
                    # token_text = html.unescape(split[token_index])
                    token_text_original = split[token_index]
                    token_text = fully_unescape_token(token_text_original)
                    begin_index = last_token_end_index  # + 1
                    end_index = last_token_end_index + len(token_text)  # + 1

                    # if token_text[0] in PUNCTUATION_AND_OTHERS:
                    #    begin_index -= 1
                    #    end_index -= 1

                    h_token = HorusToken(text=token_text,
                                         text_original=token_text_original,
                                         begin_index=begin_index,
                                         end_index=end_index,
                                         ner=split[ner_index].replace('\r', '').replace('\n', ''))

                    horus_sent.add_token(h_token)
                    last_token_end_index = h_token.end_index

                else:
                    horus_sent.get_sentence_no_space()
                    horus.add_sentence(horus_sent)
                    tot_sentences += 1
                    horus_sent = HorusSentence(index=tot_sentences, text=None, tokens=[])
                    last_token_end_index = 0  # -1

            if len(horus_sent.tokens) > 0:
                horus_sent.get_sentence_no_space()
                horus.add_sentence(horus_sent)

        config.logger.info('%s sentences processed successfully' % str(len(horus.sentences)))
        # return sentences
        return horus

    except Exception as e:
        raise e


def sent2horus(dspath: str, dataset: str, token_index: int = 0, ner_index: int = 1, separator: str = '\t',
               language: str = 'EN', tools: NLPTools = None, tokenizer_id: int = definitions.POS_TAGGER_TWITTERNLP):
    try:
        horus = Horus(dataset=dataset, language=language)
        import codecs
        sentences = []
        tokens = []
        tags_ner_y = []
        s = ''
        # has3NER = -1
        tot_sentences = 1
        last_end_index = -1
        config.logger.info('processing sentences...')
        horus_sent = HorusSentence()
        with open(dspath) as f:
            docstart = False
            for line in f:
                if line.strip() != '':
                    if separator == '': separator = None
                    splited = line.split(separator)
                    if len(splited) < 2:  # solving a specific bug
                        token = '.'
                        ner = 'O'
                        config.logger.error('attention!!!! this can not happen more than once or twice...')
                    else:
                        token = splited[token_index]
                        ner = splited[ner_index].replace('\r', '').replace('\n', '')
                if line.strip() == '':
                    if docstart is False:
                        if len(tokens) != 0:
                            config.logger.debug('processing sentence %s' % str(tot_sentences))

                            # do a word-level tokenization to duplicate labels and guarantee we always have
                            # tot-tokens >= new tokenizer number of tokens, thus, making sure we reach an alignment
                            # tokenized_sentence, labels = \
                            #    word_level_tokenization_to_preserve_labels(tokens, tags_ner_y, tools)

                            # get POS for the final sentence
                            _tokens_twe, _pos_twe, _pos_uni_twe, _probs = tokenize_and_pos(sentence=s,
                                                                                           annotator_id=tokenizer_id,
                                                                                           tools=tools)

                            _pos_twe = numpy.array(_pos_twe)
                            _pos_uni_twe = numpy.array(_pos_uni_twe)
                            assert (len(_tokens_twe) == len(tags_ner_y))
                            comp_twe = __get_compounds(_pos_twe)

                            # update the horus object
                            for index in range(0, len(_pos_twe)):
                                horus_sent.tokens[index].label_pos = _pos_twe[index]
                                horus_sent.tokens[index].label_pos_prob = _probs[index]
                                if index == 0:
                                    horus_sent.tokens[index].begin_index = 0
                                else:
                                    horus_sent.tokens[index].begin_index = \
                                        horus_sent.tokens[index - 1].end_index + 1
                                horus_sent.tokens[index].end_index = horus_sent.tokens[index].begin_index + \
                                                                     len(horus_sent.tokens[index].token) + 1

                            # verify if loop sentence == constructed sentence by tokenizer
                            assert horus_sent.get_sentence() == s

                            # set index
                            horus_sent.set_index(tot_sentences)
                            horus.add_sentence(horus_sent)

                            # sentences.append(process_sentence(has3NER, s, tokens, tags_ner_y, tools))
                            tokens = []
                            tags_ner_y = []
                            s = ''
                            # has3NER = -1
                            tot_sentences += 1
                            if tot_sentences % 100 == 0:
                                config.logger.info('caching sentence: %s' % str(tot_sentences))
                            # new object
                            horus_sent = HorusSentence()
                    else:
                        docstart = False
                else:
                    if token != '-DOCSTART-':
                        # TODO: verificar
                        if token[0] != '\'':
                            s += token + ' '
                        else:
                            s = s.strip()
                            s += token + ' '
                        beg_index = last_end_index + 1
                        end_index = beg_index + len(token)
                        horus_token = HorusToken(text=token, begin_index=beg_index, end_index=end_index, ner=ner)
                        horus_sent.add_token(horus_token)
                        last_end_index = end_index
                        tokens.append(token)
                        tags_ner_y.append(ner)
                        # if ner in definitions.NER_TAGS:
                        #    has3NER = 1
                    else:
                        docstart = True

        config.logger.info('%s sentences processed successfully' % str(len(sentences)))
        # return sentences
        return horus
    except Exception as error:
        config.logger.error('caught this error: ' + repr(error))


def conll2horus(ds: [], tokenindex: int = 0, ner_index: int = 1, sep: str = '\t', config: HorusConfig = None,
                tools: NLPTools = None, force=False):
    try:

        conll_file = ds[1] + ds[2]
        conll_label = ds[0]

        if conll_file is None or conll_label is None:
            raise Exception("Provide an input file format to be annotated")

        assert '.horusx' in conll_file

        # this is the final name after exporting the data to horus at stage 1
        horus_file_stage1_simple_json = conll_file.replace('.horusx', '.horus1.simple.json')
        horus_file_stage1 = conll_file.replace('.horusx', '.horus1.json')

        # replace the label (delete) to get the original conll file name
        conll_file = conll_file.replace('.horusx', '')
        config.logger.info('conll2horus file -> %s' % conll_file)

        if os.path.isfile(horus_file_stage1) and force is False:
            config.logger.info("file %s is already cached!" % horus_file_stage1)
        else:
            config.logger.info('exporting *.horus1.json')
            config.logger.info(' -- setting up basic structure')
            horus = conll2horus_transform_basic_structure(dspath=conll_file, dataset=conll_label,
                                                          token_index=tokenindex, ner_index=ner_index, separator=sep)

            config.logger.info(' -- adding POS tags and performing labels alignment')
            horus = add_pos_tags_to_horus_metadata(horus, tools=tools)

            config.logger.info(' -- done! saving files')
            HorusDataLoader.save_metadata_to_file(horus=horus, file=horus_file_stage1_simple_json, simple_json=True)
            HorusDataLoader.save_metadata_to_file(horus=horus, file=horus_file_stage1, simple_json=False)

            config.logger.info('file %s has been processed successfully!' % horus_file_stage1)

            # if len(sent_tokenize_list) > 0:
            #    config.logger.info('exporting *.horus2')
            #    horus_matrix = __get_horus_matrix_structure(sent_tokenize_list)
            #    __download_and_cache_results(horus_matrix)
            #    writer = csv.writer(open(file_horus_matrix, 'wb'), dialect="excel", delimiter=sep,
            #                        skipinitialspace=True)
            #    writer.writerow(definitions.HORUS_MATRIX_HEADER)
            #    writer.writerows(horus_matrix)
            #    config.logger.info('process finished! horus_matrix exported: ' + file_horus_matrix)
            # else:
            #    config.logger.warn('well, nothing to do today...')

    except Exception as e:
        config.logger.error(repr(e))


if __name__ == '__main__':

    config = HorusConfig()
    tools = NLPTools(config)

    # initialize the horus metadata file for each dataset
    for ds in definitions.NER_DATASETS:
        try:
            conll2horus(ds=ds, tokenindex=0, ner_index=1, sep='\t', config=config, tools=tools, force=True)
            config.logger.info('---')
        except Exception as e:
            config.logger.error(str(e))
            continue

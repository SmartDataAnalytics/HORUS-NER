from config import HorusConfig
from src import definitions
from src.definitions import PRE_PROCESSING_STATUS
from src.horus_meta import HorusDataLoader, Horus
from src.utils.search_engines import query_bing, query_flickr, query_wikipedia
from src.utils.sqlite_helper import *
import pandas as pd
from time import gmtime, strftime
from django.core.exceptions import ValidationError
from django.core.validators import URLValidator
import requests


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


def cache_images_and_news(horus: Horus):
    try:
        with SQLiteHelper(config.database_db) as sqlcon:
            config.logger.info('caching results...')
            t = HorusDB(sqlcon)
            auxc = 1
            download = False
            nr_tokens = 0
            for s in horus.sentences:
                for t in s.tokens:
                    if t.label_pos in definitions.POS_NOUN_TAGS or t.is_compound == 1:
                        nr_tokens += 1

            try:
                # getting list of cached terms
                values = (config.search_engine_api, config.search_engine_features_text)
                df = pd.read_sql_query(sql=SQL_ALL_TERM_SEARCH_SEL, con=t.conn, params=values)
                df.set_index("id", inplace=True)
                for s in range(0, len(horus.sentences)):
                    for t in range(0, len(horus.sentences[s].tokens)):
                        token = horus.sentences[s].tokens[t]
                        term = token.text.lower()
                        if (token.label_pos in definitions.POS_NOUN_TAGS) or token.is_compound == 1:
                            if auxc % 1000 == 0:
                                config.logger.debug('caching token %s - %s [%s]' % (str(auxc), str(nr_tokens), term))
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
                                    metaquery, result_txts, result_imgs = \
                                        query_bing(term,
                                                   key=config.search_engine_key,
                                                   top=config.search_engine_tot_resources)
                                # Flickr
                                elif int(config.search_engine_api) == 3:
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
                                horus.sentences[s].tokens[t].features.text_features.id = id_term_search
                                seq = 0
                                for web_result_txt in result_txts:
                                    config.logger.info('caching (web site) -> [%s]' % web_result_txt['displayUrl'])
                                    seq += 1
                                    t.save_website_data(id_term_search, seq, web_result_txt['id'],
                                                        web_result_txt['displayUrl'],
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
                                horus.sentences[s].tokens[t].features.image_features.id = id_term_img
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
                                                      web_result_img['width'], web_result_img['thumbnailUrl'],
                                                      str(auxtype))
                                # t.commit()
                                # adding the new item to the cache dataframe
                                config.logger.debug('updating local cache  ...')
                                cols = ['id', 'term', 'id_search_type', 'tot_results_returned']
                                df_txt = pd.DataFrame([[id_term_search, term, 1, len(result_txts)]], columns=cols)
                                df_txt.set_index("id", inplace=True)
                                df_img = pd.DataFrame([[id_term_img, term, 2, len(result_imgs)]], columns=cols)
                                df_img.set_index("id", inplace=True)
                                df = pd.concat([df, df_txt, df_img], ignore_index=False, verify_integrity=True)
                                config.logger.debug('OK')

                                # df.conca(df_txt, ignore_index=False, verify_integrity=True)
                                # df=df.append(df_img, ignore_index=False, verify_integrity=True)

                            else:
                                if len(res) != 2:
                                    raise Exception("that should not happen! check db integrity")

                                if 1 in set(df.loc[(df['term'] == term)]['id_search_type']):
                                    horus.sentences[s].tokens[t].features.text_features.id = \
                                        int(df.loc[(df['term'] == term) & (df['id_search_type'] == 1)].index.values)
                                else:
                                    horus.sentences[s].tokens[t].features.text_features.id = -1

                                if 2 in set(df.loc[(df['term'] == term)]['id_search_type']):
                                    horus.sentences[s].tokens[t].features.image_features.id = \
                                        int(df.loc[(df['term'] == term) & (df['id_search_type'] == 2)].index.values)
                                else:
                                    horus.sentences[s].tokens[t].features.image_features.id = -1

                        auxc += 1

                    if download:
                        t.commit()
                return horus

            except Exception as e:
                try:
                    if download:
                        t.commit()
                except Exception:
                    pass
                raise e
    except Exception as e:
        config.logger.error(repr(e))
        raise e


if __name__ == '__main__':

    config = HorusConfig()

    # initialize the horus metadata file for each dataset
    for ds in definitions.NER_DATASETS:
        try:
            conll_file = ds[1] + ds[2]
            assert '.horusx' in conll_file
            horus_file_stage1 = conll_file.replace('.horusx', '.horus1.json')

            config.logger.info(' loading horus file')
            horus = HorusDataLoader.load_metadata_from_file(file=horus_file_stage1)

            config.logger.info(' caching')
            # cache images and news and update the horus metadata with database ids.
            cache_images_and_news(horus)

            horus.update_status(PRE_PROCESSING_STATUS["CACHE"])

            config.logger.info(' -- done! saving files')

            horus_file_stage2_simple_json = conll_file.replace('.horusx', '.horus2.simple.json')
            horus_file_stage2 = conll_file.replace('.horusx', '.horus2.json')

            # TODO: for now I am saving in a different json file just to compare and check things are fine.
            # later just update the status of the horus file (definitions.PRE_PROCESSING_STATUS)
            HorusDataLoader.save_metadata_to_file(horus=horus, file=horus_file_stage2_simple_json, simple_json=True)
            HorusDataLoader.save_metadata_to_file(horus=horus, file=horus_file_stage2, simple_json=False)

            config.logger.info('---')

        except Exception as e:
            config.logger.error(str(e))
            continue

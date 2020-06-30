from config import HorusConfig
from src import definitions
from src.definitions import PRE_PROCESSING_STATUS
from src.features.horus_feature_extraction import HorusExtractorImage, HorusExtractorLexical, HorusExtractorText
from src.horus_meta import HorusDataLoader


if __name__ == '__main__':

    config = HorusConfig()

    EXTRACT_LEXICAL = False
    EXTRACT_TEXT = True
    EXTRACT_IMAGE = False

    fe_lexical = None
    fe_text = None
    fe_image = None

    if EXTRACT_LEXICAL:
        fe_lexical = HorusExtractorLexical(config)
    if EXTRACT_TEXT:
        fe_text = HorusExtractorText(config)
    if EXTRACT_IMAGE:
        fe_image = HorusExtractorImage(config)

    # initialize the horus metadata file for each dataset
    for ds in definitions.NER_DATASETS:
        try:
            conll_file = ds[1] + ds[2]
            assert '.horusx' in conll_file
            horus_file_stage2 = conll_file.replace('.horusx', '.horus2.json')

            config.logger.info('loading horus file: ' + horus_file_stage2)
            horus = HorusDataLoader.load_metadata_from_file(file=horus_file_stage2)

            if EXTRACT_LEXICAL and (str(PRE_PROCESSING_STATUS["FEATURE_LEXICAL"]) not in str(horus.processing_status)):
                config.logger.info('feature extraction (lexical)')
                out = fe_lexical.extract_features(horus)
                config.logger.info(f'finish ok?: {out}')
                horus.update_status(PRE_PROCESSING_STATUS["FEATURE_LEXICAL"])
            else:
                config.logger.info('feature extraction (lexical): either not active or already processed')

            if EXTRACT_IMAGE and (str(PRE_PROCESSING_STATUS["FEATURE_IMAGE"]) not in str(horus.processing_status)):
                config.logger.info('feature extraction (image)')
                out = fe_image.extract_features(horus)
                config.logger.info(f'finish ok?: {out}')
                horus.update_status(PRE_PROCESSING_STATUS["FEATURE_IMAGE"])
            else:
                config.logger.info('feature extraction (image): either not active or already processed')

            if EXTRACT_TEXT and (str(PRE_PROCESSING_STATUS["FEATURE_TEXT"]) not in str(horus.processing_status)):
                config.logger.info('feature extraction (text)')
                out = fe_text.extract_features(horus)
                config.logger.info(f'finish ok?: {out}')
                horus.update_status(PRE_PROCESSING_STATUS["FEATURE_TEXT"])
            else:
                config.logger.info('feature extraction (text): either not active or already processed')

            config.logger.info('done! saving files')
            horus_file_stage3_simple_json = conll_file.replace('.horusx', '.horus3.simple.json')
            horus_file_stage3 = conll_file.replace('.horusx', '.horus3.json')

            # TODO: for now I am saving in a different json file just to compare and check things are fine.
            # later just update the status of the horus file (definitions.PRE_PROCESSING_STATUS)
            HorusDataLoader.save_metadata_to_file(horus=horus, file=horus_file_stage3_simple_json, simple_json=True)
            HorusDataLoader.save_metadata_to_file(horus=horus, file=horus_file_stage3, simple_json=False)

            config.logger.info('hooray!')

        except Exception as e:
            config.logger.error(str(e))
            continue

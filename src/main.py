# -*- coding: utf-8 -*-

"""
==========================================================
HORUS: Named Entity Recognition Algorithm
==========================================================

HORUS is a Named Entity Recognition Algorithm specifically
designed for short-text, i.e., microblogs and other noisy
datasets existing on the web, e.g.: social media, some web-
sites, blogs and etc..

It is a multi-level machine learning framework
which combines computer vision and text mining
techniques to boost NLP tasks.

more info at: http://horus-ner.org/

"""

# Author: Esteves <diegoesteves@gmail.com>
# License: Apache License
from optparse import OptionParser

from src.config import HorusConfig
from src.core.feature_extraction.features import FeatureExtraction


def main():

    op = OptionParser(usage='usage: %prog [options] arguments (example: main.py --text="paris hilton was once the toast of the town."')

    op.add_option("--text", dest="text", help="The text to be annotated")
    op.add_option("--file", dest="file", help="The file to be annotated")
    op.add_option("--ds_format", dest="ds_format", default=0, help="The format to be annotated [0 = input text (default), 1 = Ritter, 2 = CoNNL]")
    op.add_option("--output_file", dest="output_file", default="horus_out", help="The output file")
    op.add_option("--output_format", dest="output_format", default="json", help="The output file type")

    (opts, args) = op.parse_args()
    print(__doc__)
    op.print_help()

    if not opts.text and not opts.file:
        op.error('inform either an [text] or [file] as parameter!')

    config = HorusConfig()
    extractor = FeatureExtraction(config, load_sift=1, load_tfidf=1, load_cnn=0, load_topic_modeling=1)
    print(extractor.extract_features_from_text(opts.text))


if __name__ == '__main__':
    main()

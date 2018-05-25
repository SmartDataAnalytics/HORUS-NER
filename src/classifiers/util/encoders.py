import os
import pickle

from src.core.util import definitions


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Encoders(object):
    __metaclass__ = Singleton

    def __init__(self):
        self.enc_words = None
        self.enc_stem = None
        self.enc_lemma = None

        if os.path.isfile(definitions.encoder_words):
            with open(definitions.encoder_words, 'rb') as input:
                self.enc_words = pickle.load(input)

        if os.path.isfile(definitions.encoder_stem_name):
            with open(definitions.encoder_stem_name, 'rb') as input:
                self.enc_stem = pickle.load(input)

        if os.path.isfile(definitions.encoder_lemma_name):
            with open(definitions.encoder_lemma_name, 'rb') as input:
                self.enc_lemma = pickle.load(input)

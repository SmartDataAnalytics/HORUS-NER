# -*- coding: utf8 -*-
import urllib
from urlparse import urlparse

from xml.etree import ElementTree

import re
import requests
from horus.core.config import HorusConfig
from horus.core.translation.auth import AzureAuthClient


class BingTranslator(object):

    def __init__(self, config):
        self.headers = {'Accept': 'application/xml'}
        self.detectUrl = "https://api.microsofttranslator.com/V2/Http.svc/Detect"
        self.translateUrl = "https://api.microsofttranslator.com/v2/Http.svc/Translate"
        self.config = config

    def get_token(self):
        auth_client = AzureAuthClient(self.config.translation_secret)
        return 'Bearer ' + auth_client.get_access_token()

    def clean_text(self, text):
        return re.sub('\W+',' ', text)

    def detect_language(self, text):
        try:
            text = self.clean_text(text)
            if len(str(text)) == 0:
                return 'en'
            if isinstance(text, unicode) == True:
                text = text.encode('ascii','ignore')
            params = {'appid': self.get_token(), 'text': text}
            translationData = requests.get(self.detectUrl, params=params ,headers=self.headers)
            if translationData.status_code != 200:
                raise Exception(':: error: bing lang detection status code: ' + str(translationData.status_code) + ' - ' + str(translationData.text))
            translation = ElementTree.fromstring(translationData.text.encode('utf-8'))
            return translation.text
        except Exception as e:
             raise e

    def translate(self, text, to_lang):
        try:
            text = self.clean_text(text)
            if len(str(text)) == 0:
                return ''
            if isinstance(text, unicode) == True:
                text = text.encode('ascii','ignore')

            params = {'appid': self.get_token(), 'text': text, 'to': to_lang}
            translationData = requests.get(self.translateUrl, params=params, headers=self.headers) #urllib.urlencode()
            if translationData.status_code != 200:
                raise Exception(':: error: bing translation status code: ' + str(translationData.status_code) + ' - ' +
                                str(translationData.text))
            translation = ElementTree.fromstring(translationData.text.encode('utf-8'))
            return translation.text
        except:
            raise

if __name__ == "__main__":
    config = HorusConfig()
    t = BingTranslator(config)
    print t.translate("hey what's up dude?", 'pt-br')
    print t.detect_language("Que lingua estou falando, amigo?")
    print t.detect_language("Green Newsfeed")
    print t.translate("Green Newsfeed", 'pt')
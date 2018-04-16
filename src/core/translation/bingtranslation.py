# -*- coding: utf8 -*-

from xml.etree import ElementTree

import re
import requests
from config import HorusConfig
from src.core.translation.auth import AzureAuthClient


class BingTranslator(object):

    def __init__(self, config):
        self.detectUrl = "https://api.microsofttranslator.com/V2/Http.svc/Detect"
        self.translateUrl = "https://api.microsofttranslator.com/v2/Http.svc/Translate"
        self.config = config

    def get_header(self):
        auth_client = AzureAuthClient(self.config.translation_secret)
        token = 'Bearer ' + auth_client.get_access_token()
        return {'Accept': 'application/xml', 'Authorization': token}

    def clean_text(self, text):
        return re.sub('\W+',' ', text)

    def detect_language(self, text):
        try:
            text = self.clean_text(text)
            if len(str(text)) == 0:
                return 'en'
            if isinstance(text, unicode) == True:
                text = text.encode('ascii','ignore')
            params = {'text': text}
            h = self.get_header()
            print(h)
            translationData = requests.get(self.detectUrl, params=params ,headers=h)
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

            params = {'text': text, 'to': to_lang}
            h = self.get_header()
            print(h)
            translationData = requests.get(self.translateUrl, params=params, headers=h) #urllib.urlencode()
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
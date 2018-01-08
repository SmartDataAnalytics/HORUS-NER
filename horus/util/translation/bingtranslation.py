from xml.etree import ElementTree

import requests

from horus.core.config import HorusConfig
from horus.util.translation.auth import AzureAuthClient


class BingTranslator(object):

    def __init__(self):
        config = HorusConfig()
        auth_client = AzureAuthClient(config.translation_secret)
        self.final_token = 'Bearer ' + auth_client.get_access_token()

    def translate(self, input_text, to_lang):
        headers = {"Authorization ": self.final_token}
        translateUrl = "http://api.microsofttranslator.com/v2/Http.svc/Translate?text={}&to={}".format(input_text, to_lang)
        translationData = requests.get(translateUrl, headers=headers)
        translation = ElementTree.fromstring(translationData.text.encode('utf-8'))
        return translation.text

if __name__ == "__main__":
    t = BingTranslator()
    print t.translate("hey what's up dude?", 'pt-br')

# -*- coding: utf8 -*-
# import logging
import urllib2

import re
import urllib3
import urllib
import requests
from xml.etree import ElementTree
#sess = requests.Session()
#adapter = requests.adapters.HTTPAdapter(max_retries = 20)
#sess.mount('http://', adapter)
from requests.auth import HTTPBasicAuth
import json

from horus.core.config import HorusConfig
import time
import unidecode

detect_url = "https://api.microsofttranslator.com/V2/Http.svc/Detect"
translate_url = "https://api.microsofttranslator.com/v2/Http.svc/Translate"

host = 'api.microsofttranslator.com'
path_translate = '/V2/Http.svc/Translate'


def clean_text(text):
    if isinstance(text, unicode) == True:
        text = text.encode('ascii', 'ignore')
    return re.sub('\W+', ' ', text)

def bing_translate_text(text, to, key):
    try:
        new = clean_text(text)
        params = 'to=' + to + '&text=' + new
        headers = {'Ocp-Apim-Subscription-Key': key}
        response = requests.get(translate_url, params=params, headers=headers)
        if response.status_code != 200:
            raise Exception(':: bing translation: ' + str(response.status_code) + ' - ' + str(response.text))
        translation = ElementTree.fromstring(response.text.encode('utf-8'))
        return translation.text
    except:
        raise

def bing_detect_language(text, key):
    try:
        text = clean_text(text)
        params = {'text': text}
        headers = {'Ocp-Apim-Subscription-Key': key}
        response = requests.get(detect_url, params=params, headers=headers)
        if response.status_code != 200:
            raise Exception(':: bing lang detection: ' + str(response.status_code) + ' - ' + str(response.text))
        translation = ElementTree.fromstring(response.text.encode('utf-8'))
        return translation.text
    except:
        raise

if __name__ == "__main__":
    config = HorusConfig()
    for i in range(100):
        print(bing_detect_language('ola tomas tudo bem?', config.translation_secret) + ' - ' + str(i))
        print(bing_translate_text('sim sim por aqui tudo bem e com voce? Hup &%5', 'asdasdas', config.translation_secret) + ' - ' + str(i))
        time.sleep(3)
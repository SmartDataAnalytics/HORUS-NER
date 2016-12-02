from horus.main import horus

import json
import requests
import urllib
args = {
        'client_id': 'horusclientid',#your client id here
        'client_secret': 'BljqOjqstrr4Klv9VyRwAL3Fg5bOquN8rmICYZ0qsd8=',#your azure secret here
        'scope': 'http://api.microsofttranslator.com',
        'grant_type': 'client_credentials'
    }
oauth_url = 'https://datamarket.accesscontrol.windows.net/v2/OAuth2-13'
oauth_junk = json.loads(requests.post(oauth_url, data=urllib.urlencode(args)).content)
translation_args = {
        'text': "agora nao da mais problema?",
        'to': 'en'
        }
headers={'Authorization': 'Bearer '+oauth_junk['access_token']}
translation_url = 'http://api.microsofttranslator.com/V2/Ajax.svc/Translate?'
translation_result = requests.get(translation_url+urllib.urlencode(translation_args), headers=headers)
print translation_result.content


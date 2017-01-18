import logging
import urllib2

import urllib3
import urllib
import requests
#sess = requests.Session()
#adapter = requests.adapters.HTTPAdapter(max_retries = 20)
#sess.mount('http://', adapter)
from requests.auth import HTTPBasicAuth
import json

# json_result['d']['results']
# Each result has attributes like 'Description', 'Title', 'Url', '__metadata',
# 'DisplayUrl' and 'ID' so you can do what you want!


def bing_api2(query, api, source_type="Web", top=10, format='json', market='en-US'):

    try:
        keyBing = api  # get Bing key from: https://datamarket.azure.com/account/keys
        credentialBing = 'Basic ' + (':%s' % keyBing).encode('base64')[
                                    :-1]  # the "-1" is to remove the trailing "\n" which encode adds
        query = '%27' + urllib.quote(query) + '%27'
        market = '%27' + urllib.quote(market) + '%27'
        offset = 0

        url = 'https://api.datamarket.azure.com/Bing/Search/' + source_type + \
              '?Query=%s&Market=%s&$top=%d&$skip=%d&$format=json' % (query, market, int(top), offset)

        request = urllib2.Request(url)
        request.add_header('Authorization', credentialBing)
        requestOpener = urllib2.build_opener()
        response = requestOpener.open(request)

        results = json.load(response)

        if response.code != 200:
            return query, None
        else:
            return query, results

    except Exception as e:
        return query, None
        #logging.error(':: an error has occurred: ', e)
        #raise


def bing_api(query, api, source_type="Web", top=10, format='json', market='en-US'):
    """Returns the decoded json response content
    :param query: query for search
    :param source_type: type for seacrh result
    :param top: number of search result
    :param format: format of search result
    :param market: market of search result
    """
    try:
        # set search url
        query = '%27' + urllib.quote(query) + '%27'
        market2 = '%27' + urllib.quote(market) + '%27'
        # web result only base url
        base_url = 'https://api.datamarket.azure.com/Bing/Search/' + source_type
        url = base_url + '?Query=' + query + '&Market=' + market2 + '&$top=' + str(top) + '&$format=' + format

        # create credential for authentication
        user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_3) AppleWebKit/537.36 (KHTML, like Gecko) " \
                     "Chrome/42.0.2311.135 Safari/537.36"

        urllib3.disable_warnings()
        http = urllib3.PoolManager(num_pools=10)
        headers = urllib3.util.make_headers(user_agent=user_agent, basic_auth='{}:{}'.format(api, api))
        resp = http.request('GET',
                            url=url,
                            headers=headers)
        jsonobject = json.loads(resp.data.decode('utf-8'))

        if resp.status != 200:
            return query, None
        else:
            return query, jsonobject

        '''
        # create auth object
        auth = HTTPBasicAuth(api, api)
        # set headers
        headers = {'User-Agent': user_agent}
        # get response from search url
        response_data = sess.get(url, headers=headers, auth=auth)
        # response_data = requests.get(url, headers=headers, auth=auth)

        if response_data.status_code == 200:
            return query, response_data.json()  # decode json response content
        else:
            return query, None
        '''
    except Exception as e:
        logging.error(':: an error has occurred: ', e)
        raise

import os
import tarfile
from config import HorusConfig
from src.utils.nlp_tools import NLPTools
import pickle
import urllib.request
import shutil
from pathlib import Path


def browncluster2dict(filepath: str, filename: str, force: bool) -> bool:

    try:
        if not os.path.exists(filepath + '%s_dict.pkl' % filename) or force is True:
            config.logger.info('creating dictionary: ' + filename)
            brown = dict()
            i = 0
            with open(filepath + 'paths/' + filename) as f:
                content = f.readlines()
                for x in content:
                    i += 1
                    if i % 1000000 == 0:
                        config.logger.info('progress (lines): ' + str(i))
                    n = x.split('\t')
                    brown.update({n[1]: str(n[0])})
            with open(filepath + '%s_dict.pkl' % filename, 'wb') as output:
                pickle.dump(brown, output, pickle.HIGHEST_PROTOCOL)
            config.logger.info('ok, file generated')
        else:
            config.logger.info(filepath + '%s_dict.pkl' % filename + ' exists')
    except Exception as e:
        config.logger.error(str(e))
        raise


def get_data_brown(filepath: str, force: bool = False):

    url = 'https://s3-eu-west-1.amazonaws.com/downloads.gate.ac.uk/resources/derczynski-chester-boegh-brownpaths.tar.bz2'
    dump_file = 'derczynski-chester-boegh-brownpaths.tar.bz2'

    try:
        if not os.path.exists(filepath) or force is True:
            config.logger.info('downloading the data')
            try:
                os.makedirs(filepath)
            except FileExistsError:
                pass

            brown_file = Path(filepath + dump_file)
            if not brown_file.is_file():
                config.logger.info('brown clusters not cached, downloading...')
                with urllib.request.urlopen(url) as response, open(filepath + dump_file, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
            else:
                config.logger.info('brown clusters cached.')

            config.logger.info('extracting: ' + filepath + dump_file)
            if dump_file.endswith("tar.gz"):
                tar = tarfile.open(filepath + dump_file, "r:gz")
            elif dump_file.endswith("tar"):
                tar = tarfile.open(filepath + dump_file, "r:")
            elif dump_file.endswith("tar.bz2"):
                tar = tarfile.open(filepath + dump_file, "r:bz2")

            tar.extractall(path=filepath)
            tar.close()
            config.logger.info('extraction done.')

    except Exception as e:
        raise e


def get_data_twitter(filepath: str, force: bool = False):
    # https://www.cs.cmu.edu/~ark/TweetNLP/
    # TODO: https://www.cs.cmu.edu/~ark/TweetNLP/clusters/50mpaths2
    raise Exception('to implement')


if __name__ == '__main__':

    config = HorusConfig()
    tools = NLPTools(config)

    get_data_brown(config.dir_clusters + 'brown_clusters/', force=False)

    browncluster2dict(config.dir_clusters + 'brown_clusters/', 'gha.500M-c1000-p1.paths', force=False)
    browncluster2dict(config.dir_clusters + 'brown_clusters/', 'gha.64M-c640-p1.paths', force=False)
    browncluster2dict(config.dir_clusters + 'brown_clusters/', 'gha.64M-c320-p1.paths', force=False)

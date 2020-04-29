#!/usr/bin/python
import os
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('wordnet')

import sys
sys.path.append('/Users/diego.esteves/git/horus')

from config import HorusConfig
config = HorusConfig()

if config.resource_root_dir != '':
    print('resource dirs initialized: ', config.resource_root_dir)
else:
    print('creating default directory structure')
    try:
        os.makedirs(config.project_root_dir + "resources/datasets/")
    except FileExistsError:
        pass

    try:
        os.makedirs(config.project_root_dir + "resources/encoders/")
    except FileExistsError:
        pass

    try:
        os.makedirs(config.project_root_dir + "resources/img/")
    except FileExistsError:
        pass

    try:
        os.makedirs(config.project_root_dir + "resources/libs/")
    except FileExistsError:
        pass

    try:
        os.makedirs(config.project_root_dir + "resources/log/")
    except FileExistsError:
        pass

    try:
        os.makedirs(config.project_root_dir + "resources/models/")
    except FileExistsError:
        pass

    try:
        os.makedirs(config.project_root_dir + "resources/output/")
    except FileExistsError:
        pass

    try:
        os.makedirs(config.project_root_dir + "resources/word_clusters/")
    except FileExistsError:
        pass


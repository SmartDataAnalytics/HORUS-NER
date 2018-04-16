import os

root = os.getcwd()
'''
-- LOC
'''
PATH_TRAIN_NEG = root + "/img/loc/train/neg/"
PATH_TRAIN_POS1 = root + "/img/loc/train/pos/coast/"
PATH_TRAIN_POS2 = root + "/img/loc/train/pos/forest/"
PATH_TRAIN_POS3 = root + "/img/loc/train/pos/highway/"
PATH_TRAIN_POS4 = root + "/img/loc/train/pos/mountain/"
PATH_TRAIN_POS5 = root + "/img/loc/train/pos/open_country/"
PATH_TRAIN_POS6 = root + "/img/loc/train/pos/street/"
PATH_TRAIN_POS7 = root + "/img/loc/train/pos/suburb/"
PATH_TRAIN_POS8 = root + "/img/loc/train/pos/tall_building/"
PATH_TRAIN_POS9 = root + "/img/loc/train/pos/inside_city/"
PATH_TRAIN_POS10 = root + "/img/loc/train/pos/map2/"


PATH_TEST_NEG = root + "/img/loc/test/neg/"
PATH_TEST_POS1 = root + "/img/loc/test/pos/coast/"
PATH_TEST_POS2 = root + "/img/loc/test/pos/forest/"
PATH_TEST_POS3 = root + "/img/loc/test/pos/highway/"
PATH_TEST_POS4 = root + "/img/loc/test/pos/mountain/"
PATH_TEST_POS5 = root + "/img/loc/test/pos/open_country/"
PATH_TEST_POS6 = root + "/img/loc/test/pos/street/"
PATH_TEST_POS7 = root + "/img/loc/test/pos/suburb/"
PATH_TEST_POS8 = root + "/img/loc/test/pos/tall_building/"
PATH_TEST_POS9 = root + "/img/loc/test/pos/inside_city/"
PATH_TEST_POS10 = root + "/img/loc/test/pos/map2/"

'''
-- ORG
'''
PATH_TRAIN_ORG_POS = root + "/img/org2/train/pos/"
PATH_TRAIN_ORG_NEG = root + "/img/org2/train/neg/"
PATH_TEST_ORG_POS = root + "/img/org2/test/pos/"
PATH_TEST_ORG_NEG = root + "/img/org2/test/neg/"

'''
-- PER
'''
PATH_TRAIN_PER_POS = root + "/img/per/train/pos/"
PATH_TRAIN_PER_NEG = root + "/img/per/train/neg/"
PATH_TEST_PER_POS = root + "/img/per/test/pos/"
PATH_TEST_PER_NEG = root + "/img/per/test/neg/"


def convert_to(path, prefix):
    i = 1
    for filename in os.listdir(path):
        if filename != '.DS_Store':
            newfile = path + prefix + str(i) + filename[-4:]
            os.rename(path + filename, newfile)
            print newfile
            i += 1



#convert_to(PATH_TRAIN_POS1, 'pos-')
#convert_to(PATH_TRAIN_POS2, 'pos-')
#convert_to(PATH_TRAIN_POS3, 'pos-')
#convert_to(PATH_TRAIN_POS4, 'pos-')
#convert_to(PATH_TRAIN_POS5, 'pos-')
#convert_to(PATH_TRAIN_POS6, 'pos-')
#convert_to(PATH_TRAIN_POS7, 'pos-')
#convert_to(PATH_TRAIN_POS8, 'pos-')
#convert_to(PATH_TRAIN_POS9, 'pos-')
#convert_to(PATH_TRAIN_POS10, 'pos-')
convert_to(PATH_TRAIN_NEG, 'neg-')

#convert_to(PATH_TEST_POS1, 'pos-')
#convert_to(PATH_TEST_POS2, 'pos-')
#convert_to(PATH_TEST_POS3, 'pos-')
#convert_to(PATH_TEST_POS4, 'pos-')
#convert_to(PATH_TEST_POS5, 'pos-')
#convert_to(PATH_TEST_POS6, 'pos-')
#convert_to(PATH_TEST_POS7, 'pos-')
#convert_to(PATH_TEST_POS8, 'pos-')
#convert_to(PATH_TEST_POS9, 'pos-')
#convert_to(PATH_TEST_POS10, 'pos-')
#convert_to(PATH_TEST_NEG, 'neg-')

#convert_to(PATH_TRAIN_POS1, 'pos-')


#convert_to(PATH_TRAIN_ORG_POS, 'pos-')
#convert_to(PATH_TRAIN_ORG_NEG, 'neg-')
#convert_to(PATH_TEST_ORG_POS, 'pos-')
#convert_to(PATH_TEST_ORG_NEG, 'neg-')

#convert_to(PATH_TRAIN_PER_POS, 'pos-')
#convert_to(PATH_TRAIN_PER_NEG, 'neg-')
#convert_to(PATH_TEST_PER_POS, 'pos-')
#convert_to(PATH_TEST_PER_NEG, 'neg-')

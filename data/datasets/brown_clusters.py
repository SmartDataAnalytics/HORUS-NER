output_path='/Users/diegoesteves/Github/named-entity-recognition/horus-ner/data/datasets/'
f_out = open(output_path + 'all_sentences.txt', 'w+')
for file in ['Ritter/ner.txt',
             'wnut/2015.conll.freebase',
             'wnut/2016.conll.freebase',
             'wnut/emerging.test.annotated']:
    filepath= output_path + file
    sentence=''
    with open(filepath) as f:
        content = f.readlines()
        for x in content:
            if x!='\n':
                sentence+=x.split('\t')[0] + ' '
            else:
                f_out.write(sentence.strip() + '\n')
                sentence=''
    f_out.flush()

f_out.close()


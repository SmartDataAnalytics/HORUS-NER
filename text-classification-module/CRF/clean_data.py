import csv
import re
#
with open("/home/hady/nltk_data/corpora/conll2002/eng.train", "r") as fp:
    reader = csv.reader(fp, delimiter=' ')
    table = [row for row in reader]
    with open("/home/hady/nltk_data/corpora/conll2002/eng_new.train", "w+") as fw:
        # fieldnames = ['class', 'abstract']
        writer = csv.writer(fw, delimiter=" ")
        for row in table:
            if len(row)  > 1:
                writer.writerow([row[0], row[1] ,row[-1]])
            else:
                print("hii")
                writer.writerow([])
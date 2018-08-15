import csv
import re
#
with open("../data/dbpedia/Other_raw.csv", "r") as fp:
    reader = csv.reader(fp, delimiter=',')
    table = [row for row in reader]
    with open("../data/dbpedia/final_data/Other.csv", "w+") as fw:
        # fieldnames = ['class', 'abstract']
        writer = csv.writer(fw)
        for row in table:
            if len(row) == 2:
                writer.writerow([4, re.sub(r'[^\w\s]','',row[1])])
            else:
                print(len(row))

# tb = []
# with open("data/Organization.csv", "r") as fp:
#     reader = csv.reader(fp, delimiter='|')
#     tb = [row for row in reader]
#     for row in tb:
#         if len(row) != 2:
#             print(len(row))
#     # print x
# #
import numpy as np
import pandas as pd
CONST_WIKI_ALL = "../data/dbpedia/final_data/Other.csv"

df = pd.read_csv(CONST_WIKI_ALL)
df = df.values.tolist()

print(len(df))

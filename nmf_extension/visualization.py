from nmf_extension import *
import pandas as pd
import csv

# import data
# break it up the way we want
#
input = "../Data/GenieMessages/GenieMessages.csv"


# with open(input, newline = '') as csvfile:
#     data = csv.reader(csvfile)
#
#
#     all_messages = []
#     for row in data:
#         all_messages.append(row[1])

import csv
with open(input, encoding="utf8", errors='ignore') as f:
# with open(input, newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        #print(row)



# cols = ['Combined.messages.to.Genie_ALL']
# df = pd.read_csv(input, usecols=cols, encoding = "utf-8")
# df = pd.read_csv(input)

# # Create tfidf based on messages
# # x = df['Combined.messages.to.Genie_ALL'].values
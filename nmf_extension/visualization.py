from nmf_extension import *
import pandas as pd
import csv

# import data
# break it up the way we want
#
input = "../Data/GenieMessages/GenieMessages.csv"


with open(input) as csvfile:
    data = csv.reader(csvfile, delimiter='|')

# cols = ['Combined.messages.to.Genie_ALL']
# df = pd.read_csv(input, usecols=cols, encoding = "utf-8")
# df = pd.read_csv(input)

# # Create tfidf based on messages
# # x = df['Combined.messages.to.Genie_ALL'].values
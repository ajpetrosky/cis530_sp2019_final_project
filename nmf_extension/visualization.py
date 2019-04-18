from nmf_extension import *
import pandas as pd
import csv
from collections import Counter

# import data
# break it up the way we want
#

# need to alter the code for
input1 = "../Data/GenieMessages/GenieMessages.csv"
input2 = "../Data/Results/indMessagesDev.csv"
input3 = "../Data/Results/indMessagesTest.csv"
input4 = "../Data/Results/indMessagesTrain.csv"
input5 = "../Data/Results/nmf_dev.csv"
input6 = "../Data/Results/nmf_test.csv"
input7 = "../Data/Results/nmf_train.csv"

input = input7



# with open(input, newline = '') as csvfile:
#     data = csv.reader(csvfile)
#
#
#     all_messages = []
#     for row in data:
#         all_messages.append(row[1])



with open(input, encoding="utf8", errors='ignore') as f:
    reader = csv.reader(f)
    # Decide if we want it as one string or as one list
    all_messages = ''
    if input == input1:
        for row in reader:
            #all_messages.append(row[1])
            all_messages = all_messages + row[1]
    else:
        for row in reader:
            #all_messages.append(row[1])
            all_messages = all_messages + row[0]


    #print(all_messages)
    #print(len(all_messages))

# create a dictionary of all words
words = all_messages.split()
#print(words)

total_num_tokens = len(words)
print(total_num_tokens)  # 3,186,766

# find number of unique words
num_unique_words = len(set(words))
print(num_unique_words) # 339,574

# now print words by frequecy

freq_dict = Counter(words)
#print(freq_dict)

# length of list
n=5
# most common words
print(freq_dict.most_common(n))

# least common words
print(freq_dict.most_common()[:-n-1:-1])
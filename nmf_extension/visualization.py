from nmf_extension import *
import pandas as pd
import csv
from collections import Counter
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


# import data
# break it up the way we want
#

# need to alter the code for

# All Messages
input1 = "../Data/GenieMessages/GenieMessages.csv"

# All Messages divided into individual messages
# wait on these
input2 = "../Data/Results/indMessagesDev.csv"
input3 = "../Data/Results/indMessagesTest.csv"
input4 = "../Data/Results/indMessagesTrain.csv"

# focus on these
# Labelled Results
input5 = "../Data/Results/nmf_dev.csv"
input6 = "../Data/Results/nmf_test.csv"
input7 = "../Data/Results/nmf_train.csv"


# Dev and Train have lots of carriage returns, but test doesn't
input = input6

with open(input, 'r',errors='ignore', newline = '') as f:
    # List of lists, message then label
    results = list(csv.reader(f))

# need to remove the first row which contain headers
results = results[1:]

# Number of messages:
num_messages = len(results)

# Calculate Number of Topics
labels = [x[1] for x in results]
labels = [int(i) for i in labels]
num_topics = max(labels) + 1
#print(num_topics)

# Create List of Lists of divided messsages
# the first list will be first topic, etc.
sorted_results = [[] for x in range(num_topics)]

for row in results:
    topic = int(row[1])
    sorted_results[topic].append(row[0])

#print(sorted_results)

for i in range(num_topics):
    # Calculate number of messages in each topic
    num_in_topic = len(sorted_results[i])

    # Calculate Percentage of messages in each topic
    perc = num_in_topic/num_messages*100
    print("Topic", i, ":\t Number Messages: ", num_in_topic, "\t% of Total: ", "%.2f" % perc)

# To find frequency of each word in each topic, join all words in a given topic
#print(sorted_results[2])

# join all messages in a given topic:
for topic in sorted_results:
    print("\nNew Topic:")
    print("Token: \t\tFrequency:")
    joined_topic = ''
    for message in topic:
        joined_topic = joined_topic + " "+ message


    #___________ DATA CLEANUP START_______________#
    # Split on white space:
    words = joined_topic.split()

    # Remove punctuation:
    table = str.maketrans('', '', string.punctuation)
    words = [w.translate(table) for w in words]

    # make all words lower case:
    words = [word.lower() for word in words]

    # Remove non-alphabetic tokens
    words = [word for word in words if word.isalpha()]

    # remove stop words
    stop_words = stopwords.words('english') + ['hi', 'thank', 'genie', 'propername', 'thanks', 'like', 'city', 'rmcity', 'hello', 'i', 'you']
    words = [w for w in words if not w in stop_words]

    # ___________ DATA CLEANUP END_______________#
    # Process the frequency of words
    total_num_tokens = len(words)
    # find number of unique words
    num_unique_words = len(set(words))
    #print(num_unique_words)

    # now print words by frequency
    freq_dict = Counter(words)
    # print(freq_dict)

    # Print most common words
    # length of list
    n=5

    # most common words
    #print(freq_dict.most_common(n))
    for pair in freq_dict.most_common(n):
        print(pair[0], "\t\t", pair[1])

    # least common words
    #print(freq_dict.most_common()[:-n-1:-1])


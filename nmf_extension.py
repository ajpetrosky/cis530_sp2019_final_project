# An extension using NMF from SciKit-Learn's library

# Helpful tutorial with NMF: https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730

import pprint
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.corpus import stopwords

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--n_topics', type=int, required=True)


# Use NMF to create
def nmf_topics(input, output, n):
    cols = ['Combined.messages.to.Genie_ALL']
    df = pd.read_csv(input, usecols=cols)

    # Create tfidf based on messages
    x = df['Combined.messages.to.Genie_ALL'].values
    vect = TfidfVectorizer(max_df=0.95, min_df=0.2, max_features=n, stop_words=stopwords)
    X = vect.fit_transform(x)
    feature_names = vect.get_feature_names()

    # Run NMF on messages
    nmf_model = NMF(n_components=n, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(X)
    # W = nmf_model.fit_transform(X)
    # H = nmf_model.components_

    display_topics(nmf_model, feature_names, 10)

    #df.to_csv(path_or_buf=output, index=False)


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))


def main(args):
    nmf_topics(args.input, args.output, args.n_topics)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

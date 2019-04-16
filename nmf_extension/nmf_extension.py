# An extension using NMF from SciKit-Learn's library

# Helpful tutorial with NMF: https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.corpus import stopwords

# Use NMF to create
def nmf_topics(n):
    train_file = "../data/GenieMessagesTrain.csv"
    dev_file = "../data/GenieMessagesDev.csv"
    test_file = "../data/GenieMessagesTest.csv"

    # Train the model with train data
    cols = ['Combined.messages.to.Genie_ALL']
    df = pd.read_csv(train_file, usecols=cols)

    # Create tfidf based on messages
    x = df['Combined.messages.to.Genie_ALL'].values
    vect = TfidfVectorizer(max_df=0.9, min_df=3, max_features=160,
                           stop_words=stopwords.words('english') + ['hi', 'thank', 'genie', 'propername', 'thanks', 'like'])
    X = vect.fit_transform(x)
    feature_names = vect.get_feature_names()

    # Run NMF on messages
    nmf_model = NMF(n_components=n, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(X)
    y = nmf_model.transform(X)
    print(type(y))
    print(y.size)


    display_topics(nmf_model, feature_names, 10)

    # df.to_csv(path_or_buf=output, index=False)


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

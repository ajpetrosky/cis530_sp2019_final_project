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

    # Create TF-IDF vectorizer
    sw = stopwords.words('english') + ['hi', 'thank', 'genie', 'propername', 'thanks', 'like']
    vect = TfidfVectorizer(max_df=0.9, min_df=3, max_features=160, stop_words=sw)

    # Creating training data with vectorizer and train messages
    x = df['Combined.messages.to.Genie_ALL'].values
    X = vect.fit_transform(x)
    feature_names = vect.get_feature_names()

    # Create NMF model
    nmf_model = NMF(n_components=n, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvda').fit(X)

    # Run NMF on train data
    y = nmf_model.transform(X)
    y = np.argmax(y, axis=1)
    d = {'Combined.messages.to.Genie_ALL': x, 'Topic': y}
    results_df = pd.DataFrame(d)
    results_df.to_csv(path_or_buf='nmf_train.csv', index=False)

    # Run NMF on dev data
    df = pd.read_csv(dev_file, usecols=cols)
    x = df['Combined.messages.to.Genie_ALL'].values
    X = vect.fit_transform(x)
    y = nmf_model.transform(X)
    y = np.argmax(y, axis=1)
    d = {'Combined.messages.to.Genie_ALL': x, 'Topic': y}
    results_df = pd.DataFrame(d)
    results_df.to_csv(path_or_buf='nmf_dev.csv', index=False)

    # Run NMF on test data
    df = pd.read_csv(test_file, usecols=cols)
    x = df['Combined.messages.to.Genie_ALL'].values
    X = vect.fit_transform(x)
    y = nmf_model.transform(X)
    y = np.argmax(y, axis=1)
    d = {'Combined.messages.to.Genie_ALL': x, 'Topic': y}
    results_df = pd.DataFrame(d)
    results_df.to_csv(path_or_buf='nmf_test.csv', index=False)

    # display_topics(nmf_model, feature_names, 10)


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % topic_idx)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

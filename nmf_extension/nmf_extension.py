# An extension using NMF from SciKit-Learn's library

# Helpful tutorial with NMF: https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730
# and: https://github.com/derekgreene/topic-model-tutorial/blob/master/3%20-%20Parameter%20Selection%20for%20NMF.ipynb

import re
import csv
from gensim.models import Word2Vec
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score

# Plotting tools
import matplotlib.pyplot as plt

# 1. Wordcloud of Top N words in each topic
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors


def cloud(topic_words_list, weights_list):
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    fig, axes = plt.subplots(1, 2, figsize=(5,5), sharex=True, sharey=True)

    for i in range(len(topic_words_list)):
        words = topic_words_list[i]
        weights = weights_list[i]
        norm_weights = [float(i) / max(weights) for i in weights]
        fig.add_subplot(1, 2, i+1)
        topic_words = dict(zip(words, norm_weights))
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()


# Creates inputs for W2V
class TokenGenerator:
    def __init__( self, documents):
        self.documents = documents
        self.tokenizer = re.compile(r"(?u)\b\w\w+\b")

    def __iter__( self ):
        print("Building Word2Vec model ...")
        for doc in self.documents:
            tokens = []
            for tok in self.tokenizer.findall(doc):
                tokens.append(tok.lower())
            yield tokens


# Create W2V model
def createW2v():
    file = "../data/NormalizedStudentTeacher.csv"

    cols = ['messages']
    df = pd.read_csv(file, usecols=cols)

    # Creating training data with vectorizer and train messages
    # x1 = df['Combined.messages.to.Genie_ALL'].values.astype('U')
    # x2 = df['Combined.messages.from.Genie_ALL'].values.astype('U')
    # x = np.core.defchararray.add(x1, x2)
    x = df['messages'].values.astype('U')
    x = x.tolist()
    docs = TokenGenerator(x)
    w2v = Word2Vec(docs)
    w2v.save("w2c_model_sg.bin")


# Calculate coherence
def calculate_coherence(w2v_model, term_rankings):
    overall_coherence = 0.0
    topic_scores = []
    for topic_index in range(len(term_rankings)):
        # check each pair of terms
        pair_scores = []
        for pair in combinations(term_rankings[topic_index], 2):
            pair_scores.append(w2v_model.wv.similarity(pair[0], pair[1]))
        # get the mean for all pairs in this topic
        topic_score = sum(pair_scores) / len(pair_scores)
        topic_scores.append(topic_score)
        overall_coherence += topic_score
    # get the mean score across all topics
    return overall_coherence / len(term_rankings), topic_scores


def get_descriptor(all_terms, H, topic_index, top):
    # reverse sort the values to sort the indices
    top_indices = np.argsort(H[topic_index, :])[::-1]
    # now get the terms corresponding to the top-ranked indices
    top_terms = []
    for term_index in top_indices[0:top]:
        top_terms.append(all_terms[term_index])
    return top_terms


def topic_terms(all_terms, H, topic_index, top):
    # reverse sort the values to sort the indices
    top_indices = np.argsort(H[topic_index, :])[::-1]
    # now get the terms corresponding to the top-ranked indices
    top_terms = []
    for term_index in top_indices[0:top]:
        top_terms.append(all_terms[term_index])
    return top_terms


def topic_terms_weighted(all_terms, H, topic_index, top):
    # reverse sort the values to sort the indices
    top_indices = np.argsort(H[topic_index, :])[::-1]
    top_weights = np.sort(H[topic_index, :])[::-1]
    top_weights = top_weights.tolist()[0:top]
    # now get the terms corresponding to the top-ranked indices
    top_terms = []
    for term_index in top_indices[0:top]:
        top_terms.append(all_terms[term_index])
    return top_terms, top_weights


def probs(y):
    ps = np.sum(y, axis=1)
    maxes = np.max(y, axis=1)
    ps = np.divide(maxes, ps)
    return ps


# Use NMF to create
def nmf_topics():
    # Get w2v model
    w2v_model = Word2Vec.load("w2c_model_st_norm.bin")

    train_file = "../data/NormalizedStudentTeacher_Test.csv"
    dev_file = "../data/NormalizedStudentTeacher_Dev.csv"
    test_file = "../data/NormalizedStudentTeacher_Train.csv"

    # Train the model with train data
    cols = ['messages']
    # cols = ['Combined.messages.to.Genie_ALL', 'Combined.messages.from.Genie_ALL']
    df = pd.read_csv(train_file, usecols=cols)

    # Create TF-IDF vectorizer
    vect = TfidfVectorizer(min_df=3, max_features=5000)

    # Creating training data with vectorizer and train messages
    # x1 = df['Combined.messages.to.Genie_ALL'].values.astype('U')
    # x2 = df['Combined.messages.from.Genie_ALL'].values.astype('U')
    # x = np.core.defchararray.add(x1, x2)
    x = df['messages'].values.astype('U')
    X = vect.fit_transform(x)
    feature_names = vect.get_feature_names()

    models = []

    # for n in range(2, 16):
    #     # Create NMF model, fit it, and save
    #     nmf_model = NMF(n_components=n, init='nndsvd')
    #     W = nmf_model.fit_transform(X)
    #     H = nmf_model.components_
    #     models.append((n, W, H, nmf_model))
    #
    # n_values = []
    # coherences = []
    #
    # all_topic_scores = []
    #
    # for (n, _, H, _) in models:
    #     # Get all of the topic descriptors - the term_rankings, based on top 10 terms
    #     term_rankings = []
    #     for topic_index in range(n):
    #         term_rankings.append(topic_terms(feature_names, H, topic_index, 10))
    #     # Now calculate the coherence based on our Word2vec model
    #     n_values.append(n)
    #     c, topic_scores = calculate_coherence(w2v_model, term_rankings)
    #     coherences.append(c)
    #     all_topic_scores.append(topic_scores)
    #     print("n=%02d: Coherence=%.4f" % (n, coherences[-1]))

    # Show graph
    # limit = 16
    # start = 2
    # step = 1
    # xs = range(start, limit, step)
    # plt.plot(xs, coherences)
    # plt.xlabel("Num Topics")
    # plt.ylabel("Coherence score")
    # plt.legend("coherence_values", loc='best')
    # plt.title("Number of Topics vs. Coherence Score")
    # ymax = max(coherences)
    # xpos = coherences.index(ymax)
    # best_n = n_values[xpos]
    # plt.annotate("n=%d" % best_n, xy=(best_n, ymax), xytext=(best_n, ymax), textcoords="offset points", fontsize=16)
    # plt.show()
    # plt.savefig("./nmf_sg/co_vs_n_nmf_sg.png")

    # Focus on n=4
    nmf_model = NMF(n_components=8, init='nndsvd')
    W = nmf_model.fit_transform(X)
    H = nmf_model.components_
    best_n = 8

    # Get all of the topic descriptors - the term_rankings, based on top 10 terms
    term_rankings = []
    for topic_index in range(best_n):
        term_rankings.append(topic_terms(feature_names, H, topic_index, 10))
    # Now calculate the coherence based on our Word2vec model
    c, scores = calculate_coherence(w2v_model, term_rankings)
    # coherences.append(c)
    # all_topic_scores.append(topic_scores)
    print("n=%02d: Coherence=%.4f" % (best_n, c))

    # Save results
    # (_, _, H, nmf_model) = models[best_n]
    # scores = all_topic_scores[best_n]

    # Run NMF on train data
    y = nmf_model.transform(X)
    ps = probs(y)
    y = np.argmax(y, axis=1)
    d = {'message': x, 'topic': y, 'probability': ps}
    results_df = pd.DataFrame(d)
    results_df = results_df.sort_values('probability', ascending=False)
    results_df = results_df.sort_values('topic', kind='mergesort')
    results_df.to_csv(path_or_buf='./nmf_train_st_norm_n=' + str(best_n) + '.csv', index=False)
    sil = silhouette_score(X, y)
    print('\nSilhouette Score for nmf_train_st_norm_n=' + str(best_n) + '.csv: ' + str(sil))

    # Run NMF on dev data
    df = pd.read_csv(dev_file, usecols=cols)
    # x1 = df['Combined.messages.to.Genie_ALL'].values.astype('U')
    # x2 = df['Combined.messages.from.Genie_ALL'].values.astype('U')
    # x = np.core.defchararray.add(x1, x2)
    x = df['messages'].values.astype('U')
    X = vect.transform(x)
    y = nmf_model.transform(X)
    ps = probs(y)
    y = np.argmax(y, axis=1)
    d = {'message': x, 'topic': y, 'probability': ps}
    results_df = pd.DataFrame(d)
    results_df = results_df.sort_values('probability', ascending=False)
    results_df = results_df.sort_values('topic', kind='mergesort')
    results_df.to_csv(path_or_buf='./nmf_dev_st_norm_n=' + str(best_n) + '.csv', index=False)
    sil = silhouette_score(X, y)
    print('\nSilhouette Score for nmf_dev_st_norm_n=' + str(best_n) + '.csv: ' + str(sil))

    # Run NMF on test data
    df = pd.read_csv(test_file, usecols=cols)
    # x1 = df['Combined.messages.to.Genie_ALL'].values.astype('U')
    # x2 = df['Combined.messages.from.Genie_ALL'].values.astype('U')
    # x = np.core.defchararray.add(x1, x2)
    x = df['messages'].values.astype('U')
    X = vect.transform(x)

    test_vect = TfidfVectorizer(min_df=10, max_features=5000)
    test_vect.fit(x)
    test_feats = test_vect.get_feature_names()

    term_rankings = []
    for topic_index in range(best_n):
        l = topic_terms(feature_names, H, topic_index, 10)
        l = list(set(l) & set(test_feats))
        term_rankings.append(l)
    c, _ = calculate_coherence(w2v_model, term_rankings)
    #coherences.append(c)
    print("Test Coherence - n=%02d: Coherence=%.4f" % (best_n, c))

    y = nmf_model.transform(X)
    ps = probs(y)
    y = np.argmax(y, axis=1)
    d = {'message': x, 'topic': y, 'probability': ps}
    results_df = pd.DataFrame(d)
    results_df = results_df.sort_values(['topic', 'probability'], ascending=False, kind='mergesort')
    results_df.to_csv(path_or_buf='./nmf_test_st_norm_n=' + str(best_n) + '.csv', index=False)
    sil = silhouette_score(X, y)
    print('\nSilhouette Score for nmf_test_st_norm_n=' + str(best_n) + '.csv: ' + str(sil))

    # Print top terms per topic and their scores
    term_rankings = []
    weights = []
    for topic_index in range(best_n):
        terms, weight = topic_terms_weighted(feature_names, H, topic_index, 10)
        term_rankings.append(terms)
        weights.append(weight)
    #
    # cloud(term_rankings, weights)

    with open("./nmf_coherence_st_norm_n=" + str(best_n) + ".csv", mode="w") as f:
        cols = ['topic', 'coherence_score', 'word1', 'word2', 'word3', 'word4', 'word5', 'word6', 'word7', 'word8',
                'word9', 'word10']
        co_writer = csv.DictWriter(f, fieldnames=cols, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        co_writer.writeheader()
        rows = []
        for n in range(best_n):
            d = {'topic': n, 'coherence_score': scores[n], 'word1': term_rankings[n][0], 'word2': term_rankings[n][1],
                 'word3': term_rankings[n][2], 'word4': term_rankings[n][3], 'word5': term_rankings[n][4],
                 'word6': term_rankings[n][5], 'word7': term_rankings[n][6], 'word8': term_rankings[n][7],
                 'word9': term_rankings[n][8], 'word10': term_rankings[n][9]}
            rows.append(d)
        co_writer.writerows(rows)

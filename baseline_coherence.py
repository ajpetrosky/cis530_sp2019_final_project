import simple_baseline as sp
from gensim.models import Word2Vec
from itertools import combinations
import numpy as np
import pandas as pd
import re
import csv
from collections import defaultdict as dd

def calculate_coherence(w2v_model, term_rankings):
    overall_coherence = 0.0
    topic_scores = []
    word_vectors = w2v_model.wv
    for topic_index in range(len(term_rankings)):
        # check each pair of terms
        pair_scores = []
        for pair in combinations(term_rankings[topic_index], 2):
            if pair[0] in word_vectors.vocab and pair[1] in word_vectors.vocab:
                pair_scores.append(w2v_model.wv.similarity(pair[0], pair[1]))
        # get the mean for all pairs in this topic
        topic_score = sum(pair_scores) / len(pair_scores)
        topic_scores.append(topic_score)
        overall_coherence += topic_score
    # get the mean score across all topics
    return overall_coherence / len(term_rankings), topic_scores


def main():
    results = []

    w2v_model = Word2Vec.load("nmf_extension/w2c_model_sg.bin")
    cols = ['Combined.messages.to.Genie_ALL', 'Combined.messages.from.Genie_ALL']
    df = pd.read_csv("data/GenieMessagesTest.csv", usecols=cols)
    df["messages"] = df["Combined.messages.to.Genie_ALL"] + df["Combined.messages.from.Genie_ALL"]
    df['topic'] = np.random.randint(0, 8, df.shape[0])
    term_rankings = []
    for i in range(8):
        s = df.loc[df['topic'] == i, 'messages']
        words = dd(int)
        for (_, m) in s.iteritems():
            wordList = re.sub("[^\w]", " ", m).split()
            wordList = [w.lower() for w in wordList]
            for w in wordList:
                words[w] += 1
        top = []
        for (w, _) in sorted(words.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:100]:
            top.append(w)
        term_rankings.append(top)
    (c, r) = calculate_coherence(w2v_model, term_rankings)
    results.append(("Unnormalized Student Genie", c, r))

    w2v_model = Word2Vec.load("nmf_extension/w2c_model_sg_norm.bin")
    cols = ['messages']
    df = pd.read_csv("data/NormalizedStudentGenie_Test.csv", usecols=cols)
    df['topic'] = np.random.randint(0, 8, df.shape[0])
    term_rankings = []
    for i in range(8):
        s = df.loc[df['topic'] == i, 'messages']
        words = dd(int)
        for (_, m) in s.iteritems():
            wordList = re.sub("[^\w]", " ", str(m)).split()
            wordList = [w.lower() for w in wordList]
            for w in wordList:
                words[w] += 1
        top = []
        for (w, _) in sorted(words.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:100]:
            top.append(w)
        term_rankings.append(top)
    (c, r) = calculate_coherence(w2v_model, term_rankings)
    results.append(("Normalized Student Genie", c, r))

    w2v_model = Word2Vec.load("nmf_extension/w2c_model_st_norm.bin")
    cols = ['messages']
    df = pd.read_csv("data/NormalizedStudentTeacher_Test.csv", usecols=cols)
    df['topic'] = np.random.randint(0, 8, df.shape[0])
    term_rankings = []
    for i in range(8):
        s = df.loc[df['topic'] == i, 'messages']
        words = dd(int)
        for (_, m) in s.iteritems():
            wordList = re.sub("[^\w]", " ", str(m)).split()
            wordList = [w.lower() for w in wordList]
            for w in wordList:
                words[w] += 1
        top = []
        for (w, _) in sorted(words.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:100]:
            top.append(w)
        term_rankings.append(top)
    (c, r) = calculate_coherence(w2v_model, term_rankings)
    results.append(("Normalized Student Teacher", c, r))

    with open("./baseline_results.csv", mode="w") as f:
        cols = ['data', 'overall_coherence_score', 'coherence_score_t0', 'coherence_score_t1', 'coherence_score_t2',
                'coherence_score_t3', 'coherence_score_t4', 'coherence_score_t5', 'coherence_score_t6',
                'coherence_score_t7']
        co_writer = csv.DictWriter(f, fieldnames=cols, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        co_writer.writeheader()
        rows = []
        for n in range(len(results)):
            d = {'data': results[n][0], 'overall_coherence_score': results[n][1],
                 'coherence_score_t0': results[n][2][0], 'coherence_score_t1': results[n][2][1],
                 'coherence_score_t2': results[n][2][2], 'coherence_score_t3': results[n][2][3],
                 'coherence_score_t4': results[n][2][4], 'coherence_score_t5': results[n][2][5],
                 'coherence_score_t6': results[n][2][6], 'coherence_score_t7': results[n][2][7]}
            rows.append(d)
        co_writer.writerows(rows)


if __name__ == '__main__':
    main()


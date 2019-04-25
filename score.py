import pprint
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score


def sil_score(input_file):
    cols = ['message', 'topic']
    df = pd.read_csv(input_file, usecols=cols)
    x = df['message'].values.astype('U')
    labels = df['topic'].values.astype('int')

    cnt_vect = CountVectorizer()
    x = cnt_vect.fit_transform(x)

    score = silhouette_score(x, labels)
    print("Silhouette Score = " + str(score))

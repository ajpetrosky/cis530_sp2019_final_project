import pprint
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)


def sil_score(input):
    cols = ['Combined.messages.to.Genie_ALL', 'Topic']
    df = pd.read_csv(input, usecols=cols)
    x = df['Combined.messages.to.Genie_ALL'].values
    labels = df['Topic'].values

    cnt_vect = CountVectorizer()
    x = cnt_vect.fit_transform(x)

    score = silhouette_score(x, labels)
    print("Silhouette Score = " + str(score))




def main(args):
    sil_score(args.input)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

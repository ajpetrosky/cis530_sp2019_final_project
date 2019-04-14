import os
import pandas as pd
import pprint
import argparse
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--n_topics', type=int, required=True)


# Reads in input CSV and outputs a csv with random
def random_topic(input, output, n):
    cols = ['Combined.messages.to.Genie_ALL']
    df = pd.read_csv(input, usecols=cols)
    df['Topic'] = np.random.randint(0, n, df.shape[0])
    df.to_csv(path_or_buf=output, index=False)


def main(args):
    random_topic(args.input, args.output, args.n_topics)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

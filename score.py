import os
import csv
import pprint
import argparse
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)



def main(args):
    input = readWordPairsLabels(args.input)
    input_label_dict = random_topic(args.wikideppaths, wordpairs_labels)
    writeHypoHyperPairsToFile(hypo_hyper_pairs, args.outputfile)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

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

def create_tf_idf_matrix(term_document_matrix):
    '''Given the term document matrix, output a tf-idf weighted version.

    See section 15.2.1 in the textbook.

    Hint: Use numpy matrix and vector operations to speed up implementation.

    Input:
      term_document_matrix: Numpy array where each column represents a document
      and each row, the frequency of a word in that document.

    Returns:
      A numpy array with the same dimension as term_document_matrix, where
      A_ij is weighted by the inverse document frequency of document h.
    '''
    term_document_matrix[term_document_matrix > 0] = 1
    dfs = np.sum(term_document_matrix, axis=1)
    idfs = np.reciprocal(dfs)
    words, docs = np.shape(term_document_matrix)

    raw_idfs = np.multiply(idfs, docs)
    idfs = np.log(raw_idfs)

    tf_idf_matrix = np.zeros(np.shape(term_document_matrix))

    for word in range(words):
        for doc in range(docs):
            tf = term_document_matrix[word, doc]
            if tf > 0:
                tf = 1 + np.log(tf)
            tf_idf_matrix[word, doc] = tf * idfs[word]

    return tf_idf_matrix


def main(args):
    sil_score(args.input)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

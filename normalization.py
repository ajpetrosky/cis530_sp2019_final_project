import argparse
from ekphrasis.classes.tokenizer import SocialTokenizer, Tokenizer
from normalizer import Normalizer
from utils import import_data, split_sentences

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true')
parser.add_argument('--dev', action='store_true')
parser.add_argument('--test', action='store_true')


def get_normalizer():
    normalizer = Normalizer(
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
        #"elongated",
        annotate={"repeated", 'emphasis', 'censored'},
        remove_tags=True,
        unpack_hashtags=False,
        segmenter='english',
        spell_correction=False,
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=True).tokenize
    )

    return normalizer

def main(args):
    common = 'GenieMessages'
    if args.train:
        filename = f"{common}Train.csv"
        outfile = f"{common}TrainNormalized.csv"
    elif args.dev:
        filename = f"{common}Dev.csv"
        outfile = f"{common}DevNormalized.csv"
    elif args.test:
        filename = f"{common}Test.csv"
        outfile = f"{common}TestNormalized.csv"
    else:
        raise Exception("Train/Dev/Test selection is required.")

    data_dict = import_data(filename)
    split_dict = split_sentences(data_dict)

    norm = get_normalizer().normalize(split_dict)
    export_normalized_data(data_dict, outfile, norm)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

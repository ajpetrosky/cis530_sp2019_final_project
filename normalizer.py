import argparse
import csv
import itertools
import re
import string

from ekphrasis.classes.segmenter import Segmenter
from ekphrasis.classes.spellcorrect import SpellCorrector
from ekphrasis.classes.tokenizer import SocialTokenizer, Tokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor
from nltk.corpus import stopwords


class Normalizer(TextPreProcessor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.remove_tags = kwargs.get("remove_tags", True)

        self.tags = ['<repeated>', '<emphasis>', '<email>']

        # add custom stopwords from your dataset here
        custom_stopwords = set(['propername'])
        nltk_stopwords = set(stopwords.words('english'))

        self.stopwords = custom_stopwords.union(nltk_stopwords)

        self.seg = Segmenter()
        self.spell = SpellCorrector()

        self.elongated = self.regexes["elongated"]
        self.mini_elongated = re.compile("([a-zA-Z])\\1\\1")
        self.propername_regex = "(propername)+"
        self.repeated_digits = re.compile("\d{5,}")

    def handle_repeated_digits(self, sentence: str):
        s = re.sub(self.repeated_digits, '', sentence)
        return s

    def handle_elongated(self, sentence: str):
        s = re.sub(self.mini_elongated, '', sentence)
        s = self.elongated.sub(
            lambda w: self.handle_elongated_match(w), s)
        return s

    def handle_repeated_propername(self, sentence: str):
        regex = re.compile(self.propername_regex)
        s = re.sub(regex, '', sentence).strip()
        return s

    def correct(self, sentence: str):
        sentence = sentence.split()
        corrected = [self.spell.correct(w) if len(w) < 50 else w for w in sentence]
        return " ".join(corrected)

    def segment(self, sentence: str):
        s = sentence.split()
        segmented = []
        for tok in s:
            if len(tok) < 50:
                seg_list = self.seg.segment(tok).split()
                segmented.extend(seg_list)
            else:
                segmented.append(tok)
        segmented = list(itertools.chain(segmented))
        return " ".join(segmented)

    def strip_tags(self, sentence: list):
        if self.remove_tags:
            s = sentence
            # s = sentence.split()
            no_tags = [w for w in s if w not in self.tags]
            return " ".join(no_tags)
        else:
            return sentence

    def remove_stopwords(self, sentence: str):
        s = sentence.split()
        cleaned = [w for w in s if w.lower() not in self.stopwords]
        return " ".join(cleaned)

    def strip_punctuation(self, sentence: str):
        '''
        Remove all punctuation from the given sentence,
        including emojis like :)
        '''
        punct = string.punctuation
        s = sentence.translate(str.maketrans(punct, ' ' * len(punct)))
        return " ".join(s.split())

    def normalize(self, split_dict):
        normalized = {}
        for student in split_dict:
            cleaned = []
            for msg in split_dict[student]:
                m = msg
                m = self.pre_process_doc(m)
                m = self.strip_tags(m)
                m = self.remove_stopwords(m)
                m = self.strip_punctuation(m)
                m = self.handle_repeated_digits(m)
                m = self.handle_elongated(m)
                m = self.handle_repeated_propername(m)
                m = self.segment(m)
                m = self.correct(m)
                cleaned.append(m)
            cleaned = " | ".join(cleaned)
            normalized[student] = cleaned
        return normalized


# def main(args):
#     common = 'GenieMessages'
#     if args.train:
#         filename = f"{common}Train.csv"
#     elif args.dev:
#         filename = f"{common}Dev.csv"
#     elif args.test:
#         filename = f"{common}Test.csv"
#     else:
#         raise Exception("Train/Dev/Test selection is required.")
#
#     data_dict = import_data(filename)
#     split_dict = split_sentences(data_dict)
#
#     normalizer = Normalizer(
#         normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
#         'time', 'url', 'date', 'number'],
#         #"elongated",
#         annotate={"repeated", 'emphasis', 'censored'},
#         remove_tags=True,
#         unpack_hashtags=False,
#         segmenter='english',
#         spell_correction=False,
#         spell_correct_elong=False,
#         tokenizer=SocialTokenizer(lowercase=True).tokenize
#     )
#     normalizer.normalize(split_dict)
#
#
# if __name__ == '__main__':
#     args = parser.parse_args()
#     main(args)

import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true')
parser.add_argument('--dev', action='store_true')
parser.add_argument('--test', action='store_true')


def import_data(filename):
    data_dict = {}
    with open(filename, newline='', encoding='latin') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data_dict[row['Student_Year']] = row['Combined.messages.to.Genie_ALL']

    return data_dict

def split_sentences(data_dict):
    split_dict = {}
    for student in data_dict:
        sents = data_dict[student].split('|')
        sents = [s.strip() for s in sents]
        split_dict[student] = sents

    return split_dict

def export_normalized_data(datadict, outfile, normalized):
    with open(outfile, 'w', newline='') as csvfile:
        fieldnames = datadict.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for student in normalized:
            row = {
                'Student_Year': student,
                'Combined.messages.to.Genie_ALL': normalized[student],
                'Combined.messages.from.Genie_ALL': datadict[student]
            }
            writer.writerow(row)


def check(split_dict):
    print(split_dict['431968_131415'])


def main(args):
    common = 'GenieMessages'
    if args.train:
        filename = f"{common}Train.csv"
    elif args.dev:
        filename = f"{common}Dev.csv"
    elif args.test:
        filename = f"{common}Test.csv"
    else:
        raise Exception("Train/Dev/Test selection is required.")

    data_dict = import_data(filename)
    split_dict = split_sentences(data_dict)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

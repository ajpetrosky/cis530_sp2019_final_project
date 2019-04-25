import pprint
import argparse
import numpy as np
import pandas as pd


# Reads in input csv and outputs a csv with random label for
# each genie message
def refactor():
    f1 = "data/norm_sg/norm_first_5000_StudentGenie.csv"
    f2 = "data/norm_sg/norm_second_5000_StudentGenie.csv"
    f3 = "data/norm_sg/norm_third_5000_StudentGenie.csv"
    f4 = "data/norm_sg/norm_fourth_5000_StudentGenie.csv"
    f5 = "data/norm_sg/norm_fifth_5000_StudentGenie.csv"
    # f6 = "data/norm_sg/norm_sixth_5000_StudentGenie.csv"
    # f7 = "data/norm_sg/norm_seventh_5000_StudentGenie.csv"
    cols = ['stud', 'messages']
    dfs = []
    dfs.append(pd.read_csv(f1, usecols=cols))
    dfs.append(pd.read_csv(f2, usecols=cols))
    dfs.append(pd.read_csv(f3, usecols=cols))
    dfs.append(pd.read_csv(f4, usecols=cols))
    dfs.append(pd.read_csv(f5, usecols=cols))
    # dfs.append(pd.read_csv(f6, usecols=cols))
    # dfs.append(pd.read_csv(f7, usecols=cols))
    df = pd.concat(dfs)
    df.to_csv(path_or_buf="data/NormalizedStudentGenie_Train.csv", index=False)

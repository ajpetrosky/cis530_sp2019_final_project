import pprint
import argparse
import numpy as np
import pandas as pd


# Reads in input csv and outputs a csv with random label for
# each genie message
def refactor():
    f1 = "data/NormalizedStudentGenie.csv"
    f2 = "data/NormalizedStudentTeacherMessages.csv"

    cols = ['stud', 'messages']
    df1 = pd.read_csv(f1, usecols=cols)
    df2 = pd.read_csv(f2, usecols=cols)

    df = pd.merge(df1, df2, how='left', on='stud')
    x1 = df['messages_x'].values.astype('U')
    x2 = df['messages_y'].values.astype('U')
    x = np.core.defchararray.add(x1, x2)
    df = pd.DataFrame({'stud': df['stud'].values, 'messages': x})
    df.to_csv(path_or_buf="data/NormalizedStudentTeacher.csv", index=False)

import numpy as np
import pandas as pd


# Reads in input csv and outputs a csv with random label for
# each genie message
def random_topic(input, output, n):
    cols = ['Combined.messages.to.Genie_ALL']
    df = pd.read_csv(input, usecols=cols)
    df['Topic'] = np.random.randint(0, n, df.shape[0])
    df.to_csv(path_or_buf=output, index=False)


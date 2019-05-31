import re
import os
import multiprocessing
import langid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from constants import DF_FNAME

def drop_or_not(index, row):
    """
    Determine if current document (lyrics) should be dropped or not
    Returns index if row needs to be dropped otherwise returns -1
    """
    # Note: row['lyrics'] returns nan when empty
    lyric = '' if not row['lyrics'] else str(row['lyrics'])[:100]
    if not lyric:
        return index
    else:
        lang, _ = langid.classify(lyric)
        if lang == 'en':
            return -1
        else:
            return index

def load_data(filename):
    """
    Load data frame
    """
    if os.path.isfile(DF_FNAME) and os.path.exists(DF_FNAME):
        print('Cached dataframe found.')
        df = pd.read_pickle(DF_FNAME)
    else:
        print('Loading data...')
        df = pd.read_csv(filename)

        # Remove rows with missing values
        df.dropna()

        # Remove rows with lyrics that don't contain any letters
        df = df[df['lyrics'].str.contains('[A-Za-z]', na=False)]

        # Remove rows with non-English lyrics
        drop_indices = Parallel(n_jobs=multiprocessing.cpu_count(), prefer='threads')(delayed(drop_or_not)(index, row) for index, row in tqdm(df.iterrows(), total=df.shape[0]))
        drop_indices = [i for i in drop_indices if i >= 0]
        df = df.drop(drop_indices)

        # Cache dataframe
        df.to_pickle(DF_FNAME)

    return df

def visualize_data(df):
    """
    Take in a data frame and visualize genres vs count
    """
    # Remove 'not available'
    genres = df.genre.unique().tolist()
    remove_index = genres.index('Not Available')
    genres.pop(remove_index)
    print('Genres: ', genres)

    # Extract number of songs in each genre
    genre_counts = df.genre.value_counts().tolist()
    genre_counts.pop(remove_index)
    print('Counts: ', genre_counts)

    # Plot bar graph
    plt.bar(genres, genre_counts)
    plt.xlabel('Genres')
    plt.ylabel('Count')
    plt.show()

def main():
    df = load_data('lyrics.csv')
    # print(df.head())
    # visualize_data(df)


if __name__ == '__main__':
    main()

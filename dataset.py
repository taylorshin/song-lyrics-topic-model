import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from langdetect import detect

import multiprocessing
from joblib import Parallel, delayed

def drop_or_not(index, row):
    """
    Determine if current document (lyrics) should be dropped or not
    False -> don't drop
    True -> drop it like it's hot
    """
    lyric = '' if not row['lyrics'] else str(row['lyrics'])
    if not lyric:
        return True
    else:
        try:
            lang = detect(lyric)
            if lang is not 'en':
                return True
        except:
            print('Row {} throws error: {}'.format(index, row['lyrics']))
            return True
    return False

def drop_or_not_2(row):
    """
    Determine if current document (lyrics) should be dropped or not
    False -> don't drop
    True -> drop it like it's hot
    """
    lyric = '' if not row else str(row)
    if not lyric:
        return True
    else:
        try:
            lang = detect(lyric)
            if lang is not 'en':
                return True
        except:
            print('Row throws error: {}'.format(row))
            return True
    return False

def visualize_data(filename):
    df = pd.read_csv(filename)

    # Remove rows with missing values
    print('before removing rows with missing values: ', df.shape)
    df.dropna()
    print('after removing rows with missing values: ', df.shape)

    # TODO: Remove rows with non-English lyrics
    print('before removing non-English lyrics: ', df.shape)
    """
    For loop method
    """
    # for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    # for index, row in df.iterrows():
    #     # lyric = '' if np.isnan(row['lyrics']) else str(row['lyrics'])
    #     lyric = '' if not row['lyrics'] else str(row['lyrics'])
    #     # Drop rows with no lyrics or empty strings
    #     if not lyric:
    #         print('Empty lyric at row {}'.format(index))
    #         df.drop(index, inplace=True)
    #     else:
    #         # detect(lyric) is not 'en':
    #         try:
    #             lang = detect(lyric)
    #             if lang is not 'en':
    #                 df.drop(index, inplace=True)
    #         except:
    #             print('Row {} throws error: {}'.format(index, row['lyrics']))
    #             df.drop(index, inplace=True)
    """
    Pandas vectorization method
    """
    start = time.time()
    df[df['lyrics'].apply(drop_or_not_2) is not True]
    end = time.time()
    print('Elapsed time: ', end - start)

    """
    Parallel thread method
    """
    # start = time.time()
    # drop_indices = Parallel(n_jobs=multiprocessing.cpu_count(), prefer='threads')(delayed(drop_or_not)(index, row) for index, row in df.iterrows())
    # print('DROP INDICES: ', drop_indices)
    # end = time.time()
    # print('Elapsed time: ', end - start)

    print('after removing non-English lyrics: ', df.shape)

    genres = df.genre.unique().tolist()

    # Remove 'not available'
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
    # data = load_data('lyrics.csv')
    visualize_data('lyrics.csv')


if __name__ == '__main__':
    main()

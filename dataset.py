import re
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from langdetect import detect
from joblib import Parallel, delayed

def drop_or_not(index, row):
    """
    Determine if current document (lyrics) should be dropped or not
    Returns index if doc needs to be dropped otherwise returns -1
    """
    # Note: row['lyrics'] returns nan when empty
    lyric = '' if not row['lyrics'] else str(row['lyrics'])
    if not lyric:
        return index
    else:
        # Check if lyrics contain any letters
        if re.search('[a-zA-z]+', lyric):
            try:
                lang = detect(lyric)
                if lang is not 'en':
                    return index
            except:
                print('Row {} throws error: {}'.format(index, row['lyrics']))
                return index
        else:
            # Lyrics don't contain any letters and are probably random punctuation symbols
            return index
    return -1

def drop_or_not_2(lyric):
    """
    True -> drop it like its hot
    False -> don't drop
    """
    # try:
    #     lang = detect(lyric[:100])
    #     if lang is not 'en':
    #         return True
    # except:
    #     # print('Row {} throws error: {}'.format(index, row['lyrics']))
    #     return True
    # return False
    lang = detect(lyric[:100])
    if lang == 'en':
        return False
    else:
        return True

def load_data(filename):
    """
    Load data frame
    TODO: cache data frame if possible
    """
    df = pd.read_csv(filename)
    df = df[:100]

    # Remove rows with missing values
    df.dropna()

    # Remove rows with lyrics that don't contain any letters
    df = df[df['lyrics'].str.contains('[A-Za-z]', na=False)]

    # Remove rows with non-English lyrics
    # print('before removing non-English lyrics: ', df.shape)
    # drop_indices = Parallel(n_jobs=multiprocessing.cpu_count(), prefer='threads')(delayed(drop_or_not)(index, row) for index, row in tqdm(df.iterrows(), total=df.shape[0]))
    # drop_indices = [i for i in drop_indices if i >= 0]
    # df.drop(drop_indices)
    # print('after removing non-English lyrics: ', df.shape)
    print('before removing non-English lyrics: ', df.shape)
    drop_bool_list = np.vectorize(drop_or_not_2)(np.array(df['lyrics'].tolist()))
    print('drop bool list: ', drop_bool_list)
    drop_indices = [i for i, d in enumerate(drop_bool_list) if d]
    print('drop indices: ', drop_indices)
    df = df.drop(drop_indices)
    print('after removing non-English lyrics: ', df.shape)

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

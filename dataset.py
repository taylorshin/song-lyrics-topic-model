import re
import os
import pickle
import multiprocessing
import langid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from joblib import Parallel, delayed
from nltk.tokenize import RegexpTokenizer, TweetTokenizer
from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words
from constants import DATAFRAME_FNAME, TOKENS_FNAME, STOP_WORDS

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
    if os.path.isfile(DATAFRAME_FNAME) and os.path.exists(DATAFRAME_FNAME):
        print('Cached dataframe found.')
        df = pd.read_pickle(DATAFRAME_FNAME)
    else:
        print('Loading data...')
        df = pd.read_csv(filename)

        # Remove rows with missing values
        df = df.dropna()

        # Remove rows with lyrics that don't contain any letters
        df = df[df['lyrics'].str.contains('[A-Za-z]', na=False)]

        # Remove rows with non-English lyrics
        drop_indices = Parallel(n_jobs=multiprocessing.cpu_count(), prefer='threads')(delayed(drop_or_not)(index, row) for index, row in tqdm(df.iterrows(), total=df.shape[0]))
        drop_indices = [i for i in drop_indices if i >= 0]
        df = df.drop(drop_indices)

        # Remove songs whose year < 1970
        df = remove_old_songs(df)

        # Remove songs whose genre is 'Not Available'
        df = remove_not_available(df)

        # Cache dataframe
        df.to_pickle(DATAFRAME_FNAME)

    return df

def remove_old_songs(df, too_old=1970):
    """
    Remove the songs before 1970 and after 2019 from df
    """
    drop_indices =  df.index[df['year'] < too_old].tolist()
    df = df.drop(drop_indices)
    return df

def remove_not_available(df):
    """
    Remove the songs whose genre is Not Available
    """
    drop_indices =  df.index[df['genre'] == 'Not Available'].tolist()
    df = df.drop(drop_indices)
    return df

def remove_stop_words(stop_list, tokens):
    """
    Remove stop words from tokens and remove tokens that are single letters/char like 's'
    """
    return [t for t in tokens if len(t) > 2 and not t in stop_list]

def stem_tokens(stemmer, tokens):
    """
    Stem tokens
    """
    return [stemmer.stem(t) for t in tokens]

def remove_low_freq_tokens(freq_list, tokens):
    """
    Remove low frequency tokens
    """
    return [t for t in tokens if freq_list[t] > 1]

def tokenize_corpus(corpus):
    """
    Segment each document in corpus into words
    Also applys stop words and stemming to tokens
    """
    if os.path.isfile(TOKENS_FNAME) and os.path.exists(TOKENS_FNAME):
        print('Cached tokens found.')
        with open(TOKENS_FNAME, 'rb') as f:
            final_tokens = pickle.load(f)
    else:
        print('Tokenizing data...')
        tokens_corpus = []
        # TODO: Try the TweetTokenizer which apparently doesn't split words with apostrophes
        # tokenizer = TweetTokenizer()
        tokenizer = RegexpTokenizer(r'\w+')
        # Create English stop words list
        en_stop = get_stop_words('en') + STOP_WORDS
        # Create p_stemmer of class PorterStemmer
        p_stemmer = PorterStemmer()

        for doc in corpus:
            raw = doc.lower()
            tokens = tokenizer.tokenize(raw)
            tokens_corpus.append(tokens)

        print('Removing stop words from tokens...')
        stopped_tokens = Parallel(n_jobs=multiprocessing.cpu_count(), prefer='threads')(delayed(remove_stop_words)(en_stop, tokens) for tokens in tqdm(tokens_corpus))

        print('Stemming tokens...')
        stemmed_tokens = Parallel(n_jobs=multiprocessing.cpu_count(), prefer='threads')(delayed(stem_tokens)(p_stemmer, tokens) for tokens in tqdm(stopped_tokens))

        print('Removing low frequency tokens...')
        freq_list = defaultdict(int)
        for doc in stemmed_tokens:
            for token in doc:
                freq_list[token] += 1
        final_tokens = Parallel(n_jobs=multiprocessing.cpu_count(), prefer='threads')(delayed(remove_low_freq_tokens)(freq_list, tokens) for tokens in tqdm(stemmed_tokens))

        # Cache tokens
        with open(TOKENS_FNAME, 'wb') as f:
            pickle.dump(final_tokens, f)

    return final_tokens

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
    # visualize_data(df)
    df_list = df['lyrics'].tolist()
    tokens = tokenize_corpus(df_list)
    print(tokens[:1])
    # en_stop = get_stop_words('en') + STOP_WORDS
    # print('Stop words: ', en_stop, len(en_stop))


if __name__ == '__main__':
    main()

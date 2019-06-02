import time
import gensim
import numpy as np
import pandas as pd
from gensim import corpora, models
from dataset import load_data, tokenize_corpus

def get_feature_matrix(df):
    """
    Return feature matrix X (numpy array) for metadata (year, artist, genre)
    Note: not using song title for now
    """
    years = df.year.unique().tolist()
    artists = df.artist.unique().tolist()
    genres = df.genre.unique().tolist()

    # Encode years
    min_year = min(years)
    years_encoded = np.array((pd.to_numeric(df['year']) - min_year).tolist())

    # Encode artist
    artists_encoded = np.array((df['artist'].apply(lambda x: artists.index(x))).tolist())
    
    # Encode genre
    genres_encoded = np.array((df['genre'].apply(lambda x: genres.index(x))).tolist())

    return np.column_stack((years_encoded, artists_encoded, genres_encoded))

def main():
    # Load and tokenize data
    df = load_data('lyrics.csv')

    # Metadata feature matrix
    x = get_feature_matrix(df)

    # df_list = df['lyrics'].tolist()
    # tokens_corpus = tokenize_corpus(df_list)

    # # Convert tokenized documents into a id <-> term dictionary
    # dictionary = corpora.Dictionary(tokens_corpus)

    # # Convert tokenized documents into a document-term matrix
    # corpus = [dictionary.doc2bow(tokens_doc) for tokens_doc in tokens_corpus]

    # # Generate LDA model
    # print('Generating LDA model')
    # start = time.time()
    # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=1)
    # end = time.time()
    # print('Time elapsed: {} sec'.format(end - start))

    # print(ldamodel.print_topics(num_topics=3, num_words=3))

if __name__ == '__main__':
    main()

import time
import json
import operator
import pickle
import gensim
import dmr.dmr as dmr
import numpy as np
import pandas as pd
from gensim import corpora, models
from dataset import load_data, tokenize_corpus

def build_feature_matrix(df):
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
    # Normalize
    years_encoded = (years_encoded - np.min(years_encoded)) / np.max(years_encoded)

    # Encode artist
    artists_encoded = np.array((df['artist'].apply(lambda x: artists.index(x))).tolist())
    # Normalize
    artists_encoded = (artists_encoded - np.min(artists_encoded)) / np.max(artists_encoded)

    # Encode genre
    genres_encoded = np.array((df['genre'].apply(lambda x: genres.index(x))).tolist())
    # Normalize
    genres_encoded = (genres_encoded - np.min(genres_encoded)) / np.max(genres_encoded)

    return np.column_stack((years_encoded, artists_encoded, genres_encoded))

def init_lda(df, K=2, alpha=0.1, beta=0.01):
    # Symmetric alpha (was 0.1 before)
    # Beta was 0.01 before
    alpha = 1.0 / K
    df_list = df['lyrics'].tolist()
    corpus = tokenize_corpus(df_list)
    voca = dmr.Vocabulary()
    docs = voca.read_corpus(corpus)
    lda = dmr.LDA(K, alpha, beta, docs, voca.size())
    return corpus, voca, docs, lda

def init_dmr(df, vecs, K=2, sigma=1.0, beta=0.01):
    df_list = df['lyrics'].tolist()
    corpus = tokenize_corpus(df_list)
    voca = dmr.Vocabulary()
    docs = voca.read_corpus(corpus)
    lda = dmr.DMR(K, sigma, beta, docs, vecs, voca.size())
    return corpus, voca, docs, vecs, lda

def main():
    ### MPKATO APPROACH ###
    # Load and tokenize data
    df = load_data('lyrics.csv')

    # Metadata feature matrix
    feat_mat = build_feature_matrix(df)

    ### DMR ###
    corpus, voca, docs, vecs, lda = init_dmr(df, feat_mat)
    lda.learning(iteration=20, voca=voca)

    # Save LDA model
    with open('model_dmr.pkl', 'wb') as f:
        pickle.dump(lda, f)

    # Word probability of each topic
    wdist = lda.word_dist_with_voca(voca)

    # Save wdist to txt file
    with open('wdist_dmr.txt', 'w') as f:
        f.write(json.dumps(wdist))

    for k in wdist:
        print('TOPIC', k)
        # print("\t".join([w for w in wdist[k]]))
        # print("\t".join(["%0.2f" % wdist[k][w] for w in wdist[k]]))
        sorted_wdist_k = dict(sorted(wdist[k].items(), key=operator.itemgetter(1), reverse=True)[:20])
        for word, prob in sorted_wdist_k.items():
            print(word, prob)
        print()

    """
    ### LDA ###
    # Learning
    corpus, voca, docs, lda = nit_lda(df)
    print('Learning...')
    lda.learning(iteration=3, voca=voca)

    # Save LDA model
    with open('model_lda.pkl', 'wb') as f:
        pickle.dump(lda, f)

    # Word probability of each topic
    wdist = lda.word_dist_with_voca(voca)

    # Save wdist to txt file
    with open('wdist_lda.txt', 'w') as f:
        f.write(json.dumps(wdist))

    for k in wdist:
        print('TOPIC', k)
        # print("\t".join([w for w in wdist[k]]))
        # print("\t".join(["%0.2f" % wdist[k][w] for w in wdist[k]]))
        sorted_wdist_k = dict(sorted(wdist[k].items(), key=operator.itemgetter(1), reverse=True)[:20])
        for word, prob in sorted_wdist_k.items():
            print(word, prob)
        print()
    """

    """
    ### GENSIM APPROACH ###
    # Load and tokenize data
    df = load_data('lyrics.csv')
    df_list = df['lyrics'].tolist()
    tokens_corpus = tokenize_corpus(df_list)

    # Convert tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(tokens_corpus)

    # Convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(tokens_doc) for tokens_doc in tokens_corpus]

    # Generate LDA model
    print('Generating LDA model')
    start = time.time()
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=1)
    end = time.time()
    print('Time elapsed: {} sec'.format(end - start))

    print(ldamodel.print_topics(num_topics=2, num_words=10))
    """

if __name__ == '__main__':
    main()

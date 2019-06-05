import time
import json
import operator
import pickle
import gensim
import itertools
import multiprocessing
import dmr.dmr as dmr
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from gensim import corpora, models
from dataset import load_data, tokenize_corpus

def one_hot(i, num_classes):
    arr = np.zeros((num_classes,))
    arr[i] = 1
    return arr

def build_feature_matrix(df):
    """
    Return feature matrix X (numpy array) for metadata (year, artist, genre)
    Note: not using song title for now
    """
    years_unique = df.year.unique().tolist()
    artists_unique = df.artist.unique().tolist()
    genres_unique = df.genre.unique().tolist()

    # Encode years
    years_encoded = np.array((pd.to_numeric(df['year']) - min(years_unique)).tolist())
    # Quantize years/time into 5 year chunks
    num_year_chunks = len(years_unique) // 5
    chunk_id_list = list(itertools.chain.from_iterable(itertools.repeat(x, 5) for x in range(num_year_chunks)))
    last_chunk_id = chunk_id_list[-1]
    chunk_id_list = chunk_id_list + ([last_chunk_id] * (len(years_unique) % 5))
    # One-hot vectors of years
    years_hot = Parallel(n_jobs=multiprocessing.cpu_count(), prefer='threads')(delayed(one_hot)(chunk_id_list[year], num_year_chunks) for year in tqdm(years_encoded))
    years_hot = np.array(years_hot)

    # Encode artist
    artists_encoded = np.array((df['artist'].apply(lambda x: artists_unique.index(x))).tolist())
    # One-hot vectors of artists
    artists_hot = Parallel(n_jobs=multiprocessing.cpu_count(), prefer='threads')(delayed(one_hot)(artist, len(artists_unique)) for artist in tqdm(artists_encoded))
    artists_hot = np.array(artists_hot)

    # Encode genre
    genres_encoded = np.array((df['genre'].apply(lambda x: genres_unique.index(x))).tolist())
    # One-hot vectors of genres
    genres_hot = Parallel(n_jobs=multiprocessing.cpu_count(), prefer='threads')(delayed(one_hot)(genre, len(genres_unique)) for genre in tqdm(genres_encoded))
    genres_hot = np.array(genres_hot)

    return np.hstack((years_hot, artists_hot, genres_hot))

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
    # TODO: cache feature matrix?
    print('Building feature matrix')
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

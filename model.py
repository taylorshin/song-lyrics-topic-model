import os
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
from constants import FEATMAT_TRAIN_FNAME, FEATMAT_TEST_FNAME, TOKENS_TRAIN_FNAME, TOKENS_TEST_FNAME, OUT_DIR

def one_hot(i, num_classes):
    arr = np.zeros((num_classes,))
    arr[i] = 1
    return arr

def build_feature_matrix(df, featmat_fname):
    """
    Return feature matrix X (numpy array) for metadata (year, artist, genre)
    Note: not using song title for now
    """
    if os.path.isfile(featmat_fname) and os.path.exists(featmat_fname):
        print('Cached feature matrix found.')
        feat_mat = np.load(featmat_fname)
    else:
        print('Building feature matrix')
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
        # artists_encoded = np.array((df['artist'].apply(lambda x: artists_unique.index(x))).tolist())
        # # One-hot vectors of artists
        # artists_hot = Parallel(n_jobs=multiprocessing.cpu_count(), prefer='threads')(delayed(one_hot)(artist, len(artists_unique)) for artist in tqdm(artists_encoded))
        # artists_hot = np.array(artists_hot)

        # Encode genre
        genres_encoded = np.array((df['genre'].apply(lambda x: genres_unique.index(x))).tolist())
        # One-hot vectors of genres
        genres_hot = Parallel(n_jobs=multiprocessing.cpu_count(), prefer='threads')(delayed(one_hot)(genre, len(genres_unique)) for genre in tqdm(genres_encoded))
        genres_hot = np.array(genres_hot)

        # feat_mat = np.hstack((years_hot, artists_hot, genres_hot))
        feat_mat = np.hstack((years_hot, genres_hot))
        # Cache feature matrix
        np.save(featmat_fname, feat_mat)
    
    return feat_mat

def train_lda(corpus, voca, docs, K=10, alpha=0.1, beta=0.01, iters=20, model_fname='model_lda.pkl'):
    if os.path.isfile(model_fname) and os.path.exists(model_fname):
        print('Cached model found.')
        with open(model_fname, 'rb') as f:
            lda = pickle.load(f)
    else:
        # Symmetric alpha (was 0.1 before)
        # Beta was 0.01 before
        alpha = 1.0 / K
        lda = dmr.LDA(K, alpha, beta, docs, voca.size())
        lda.learning(iteration=iters, voca=voca)
        # Save LDA model
        with open(model_fname, 'wb') as f:
            pickle.dump(lda, f)

    return lda

def train_dmr(corpus, voca, docs, vecs, K=10, sigma=1.0, beta=0.01, iters=20, model_fname='model_dmr.pkl'):
    if os.path.isfile(model_fname) and os.path.exists(model_fname):
        print('Cached model found.')
        with open(model_fname, 'rb') as f:
            lda = pickle.load(f)
    else:
        lda = dmr.DMR(K, sigma, beta, docs, vecs, voca.size())
        lda.learning(iteration=iters, voca=voca)
        # Save LDA model
        with open(model_fname, 'wb') as f:
            pickle.dump(lda, f)

    return lda

def evaluate_lda(trained, corpus, voca, docs, vecs, K=10, alpha=0.1, beta=0.01):
    """
    Calculate perplexity score of LDA model on test data
    """
    lda = dmr.LDA(K, alpha, beta, docs, voca.size(), trained=trained)
    p_score = lda.perplexity()
    return p_score

def evaluate_dmr(trained, corpus, voca, docs, vecs, K=10, sigma=1.0, beta=0.01):
    """
    Calculate perplexity score of DMR model on test data
    """
    lda = dmr.DMR(K, sigma, beta, docs, vecs, voca.size(), trained=trained)
    p_score = lda.perplexity()
    return p_score

def print_top_words(wdist):
    """
    Print the high probability words of each topic
    """
    for k in wdist:
        print('TOPIC', k)
        # print("\t".join([w for w in wdist[k]]))
        # print("\t".join(["%0.2f" % wdist[k][w] for w in wdist[k]]))
        sorted_wdist_k = dict(sorted(wdist[k].items(), key=operator.itemgetter(1), reverse=True))
        for word, prob in sorted_wdist_k.items():
            print(word, prob)
        print()

def print_topic_probs(tdist):
    """
    Print the topic probability of each document
    """
    for first_letter in ['a', 'b', 'c', 'd', 'e']:
        for doc, vec, td in zip(corpus, vecs, tdist):
            if doc[0].startswith(first_letter):
                print('DOC', 'Words: ', doc, 'Max topic: ', np.argmax(td), 'Max prob.: ', np.max(td))
                print('ALPHA', np.dot(vec, lda.Lambda.T))

def main():
    ### MPKATO APPROACH ###
    # Load and tokenize data
    df = load_data('lyrics.csv')

    # Train test split
    df_train = df.sample(frac=0.8, random_state=42)
    df_test = df.drop(df_train.index)

    # Metadata feature matrix
    feat_mat_train = build_feature_matrix(df_train, FEATMAT_TRAIN_FNAME)
    feat_mat_test = build_feature_matrix(df_test, FEATMAT_TEST_FNAME)

    # Set up corpus, vocabulary, and documents
    df_list_train = df_train['lyrics'].tolist()
    corpus_train = tokenize_corpus(df_list_train, TOKENS_TRAIN_FNAME)
    voca_train = dmr.Vocabulary()
    docs_train = voca_train.read_corpus(corpus_train)
    df_list_test = df_test['lyrics'].tolist()
    corpus_test = tokenize_corpus(df_list_test, TOKENS_TEST_FNAME)
    voca_test = dmr.Vocabulary()
    docs_test = voca_test.read_corpus(corpus_test)

    # Parameters
    K = 10
    alpha = 0.1
    sigma=1.0
    beta=0.01

    # Create out directory
    os.makedirs(os.path.dirname('out/'), exist_ok=True)

    # Train LDA model
    model_fname = 'model_lda_k' + str(K) + '_a' + str(alpha) + '_b' + str(beta) + '.pkl'
    model_fname = os.path.join(OUT_DIR, model_fname)
    lda = train_lda(corpus_train, voca_train, docs_train, K=K, alpha=alpha, beta=beta, model_fname=model_fname)
    # Train DMR model
    # model_fname = 'model_dmr_k' + str(K) + '_s' + str(sigma) + '_b' + str(beta) + '.pkl'
    # lda = train_dmr(corpus_train, voca_train, docs_train, feat_mat_train, K=K, sigma=sigma, beta=beta, model_fname=model_fname)

    # Word probability of each topic
    wdist = lda.word_dist_with_voca(voca_train, topk=20)
    print_top_words(wdist)

    # Calculate perplexity score of LDA model
    p_score = evaluate_lda(lda, corpus_test, voca_test, docs_test, K=K, alpha=alpha, beta=beta)
    # Calculate perplexity score of DMR model
    # p_score = evaluate_dmr(lda, corpus_test, voca_test, docs_test, feat_mat_test, K=K, sigma=sigma, beta=beta)
    print('Perplexity Score:', p_score)

    # Topic probability of each document
    # tdist = lda.topicdist()

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

import time
import gensim
from gensim import corpora, models
from dataset import load_data, tokenize_corpus

def main():
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
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=1)
    end = time.time()
    print('Time elapsed: {} sec'.format(end - start))

    print(ldamodel.print_topics(num_topics=3, num_words=3))

if __name__ == '__main__':
    main()

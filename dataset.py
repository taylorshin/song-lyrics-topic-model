import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def visualize_data(filename):
    df = pd.read_csv(filename)
    # data = df.to_numpy()
    # return data
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

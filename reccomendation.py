import numpy as np
import pandas as pd
import faiss

from common import REPRESENTATION_FIELDS, create_textual_representation, embed


def prompt_for_movie():
    """Ask the user for each field and return a row dict. Blank entries are allowed."""
    row = {}
    print('Describe the movie you want recommendations for (leave blank to skip a field):')
    for key, label in REPRESENTATION_FIELDS:
        row[key] = input(f'  {label}: ').strip()
    return row


def main():
    df = pd.read_csv('netflix_dataset.csv')
    df['textual_representation'] = df.apply(create_textual_representation, axis=1)

    index = faiss.read_index('index')

    row = prompt_for_movie()
    query = create_textual_representation(row)
    print('\nSearching with:')
    print(query)

    embedding = embed(query).reshape(1, -1)
    D, I = index.search(embedding, 5)
    best_matches = np.array(df['textual_representation'])[I.flatten()]

    for match in best_matches:
        print('Next Movie')
        print(match)
        print()


if __name__ == "__main__":
    main()

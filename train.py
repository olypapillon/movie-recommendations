import numpy as np
import pandas as pd
import faiss

from common import EMBED_DIM, create_textual_representation, embed


def main():
    df = pd.read_csv('netflix_dataset.csv')
    df['textual_representation'] = df.apply(create_textual_representation, axis=1)

    index = faiss.IndexFlatL2(EMBED_DIM)
    X = np.zeros((len(df['textual_representation']), EMBED_DIM), dtype='float32')

    for i, representation in enumerate(df['textual_representation']):
        if i % 100 == 0:
            print(f'Processed {i} instances')
        X[i] = embed(representation)

    index.add(X)
    faiss.write_index(index, 'index')


if __name__ == "__main__":
    main()

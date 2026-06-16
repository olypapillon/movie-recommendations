import numpy as np
import pandas as pd
import faiss
import streamlit as st

from common import REPRESENTATION_FIELDS, create_textual_representation, embed


@st.cache_resource
def load_data():
    df = pd.read_csv('netflix_dataset.csv')
    df['textual_representation'] = df.apply(create_textual_representation, axis=1)
    index = faiss.read_index('index')
    return df, index


st.title('Movie Recommendations')
st.write('Describe a movie you like and get similar titles. Leave fields blank to skip them.')

df, index = load_data()

row = {}
for key, label in REPRESENTATION_FIELDS:
    row[key] = st.text_input(label).strip()

if st.button('Get recommendations'):
    query = create_textual_representation(row)
    if not query:
        st.warning('Please fill in at least one field.')
    else:
        with st.spinner('Finding recommendations...'):
            embedding = embed(query).reshape(1, -1)
            D, I = index.search(embedding, 6)
            matches = df.iloc[I.flatten()]
            entered_title = row['title'].lower()
            if entered_title:
                matches = matches[matches['title'].str.lower() != entered_title]
        for text in matches['textual_representation'].head(5):
            st.text(text)
            st.divider()

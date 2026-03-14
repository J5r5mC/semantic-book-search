import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data(show_spinner=False)
def load_data_and_embeddings():
    df = pd.read_csv("book.csv")
    df = df.drop(columns=['WikipediaID', 'FreebaseID', 'Author', 'Publication Date', 'Genres'], axis=1)
    vectors = np.load("book_embeddings.npy")
    return df, vectors

df, vectors = load_data_and_embeddings()
model = load_model()

index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)

st.title("Book Search App")

query = st.text_input("Enter your query:")

if query:
    query_embedding = model.encode(query)
    query_embedding = np.array(query_embedding).astype('float32')
    D, I = index.search(np.expand_dims(query_embedding, axis=0), 5)
    st.write("Top matching books:")
    for idx in I[0]:
        st.subheader(df.iloc[idx]['Title'])
        st.write(df.iloc[idx]['Summary'][:200] + '...')
        with st.expander("Full Summary"):
            st.write(df.iloc[idx]['Summary'])

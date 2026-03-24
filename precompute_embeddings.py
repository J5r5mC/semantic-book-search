import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

csv_path = "book.csv"
df = pd.read_csv(csv_path)
df = df.drop(columns=['WikipediaID', 'FreebaseID', 'Author', 'Publication Date', 'Genres'], axis=1)

# Concatenate title and summary so the embedding captures both
texts = (df['Title'] + ', ' + df['Summary']).tolist()

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

np.save("book_embeddings.npy", embeddings)
print(f"Saved {embeddings.shape[0]} embeddings to book_embeddings.npy")

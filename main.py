import pandas as pd
import numpy as np
import streamlit as st
from numpy.linalg import norm

df = pd.read_csv("joined_deduped_brokers.csv")
names = df['Company_Name'].tolist()

embeddings = np.load("embeddings_numpy.npy")

selected_name = st.selectbox("Select a name:", names)
index = names.index(selected_name)

embed = embeddings[index]

cos_sim=[]
for i in embeddings:
    cos_sim.append(np.dot(i, embed)/(norm(i)*norm(embed)))

cos_sim = np.array(cos_sim)
top_indices = np.argsort(-cos_sim)[1:21]

st.subheader("Top 20 Most Similar Companies")
for i in top_indices:
    st.write(f"**{names[i]}** : {cos_sim[i]:.4f}")
import pandas as pd
import numpy as np
import streamlit as st
from numpy.linalg import norm
import plotly.express as px

# Load data
df = pd.read_csv("joined_deduped_brokers.csv")
names = df['Company_Name'].tolist()
embeddings = np.load("embeddings_numpy.npy")  
X_embedded = np.load('X_embedded_tsne.npy') 

# Sidebar Controls
st.sidebar.header("Controls")

# Cosine similarity threshold slider
threshold = st.sidebar.slider("Cosine Similarity Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.01)

# Dropdown for number of top similar companies
top_n = st.sidebar.selectbox("Number of Top Similar Companies to Display", options=[10, 20, 30, 40, 50], index=1)

# Select a company
selected_name = st.selectbox("Select a name:", names)
index = names.index(selected_name)
embed = embeddings[index]

# Compute cosine similarities
cos_sim = np.array([np.dot(i, embed) / (norm(i) * norm(embed)) for i in embeddings])

# Sort and filter
sorted_indices = np.argsort(-cos_sim)
sorted_indices = sorted_indices[sorted_indices != index]  # Remove selected company
filtered_indices = [i for i in sorted_indices if cos_sim[i] >= threshold]
top_indices = filtered_indices[:top_n]

# Display results
st.subheader(f"Top {len(top_indices)} Similar Companies with Similarity > {threshold}")
if top_indices:
    for i in top_indices:
        st.write(f"**{names[i]}** : {cos_sim[i]:.4f}")
else:
    st.write("No companies found with similarity above the threshold.")

# === Visualization using X_embedded ===
st.subheader("Company Embeddings Visualization")

# Create a DataFrame for plotting
plot_df = pd.DataFrame(X_embedded, columns=["x", "y"])
plot_df["name"] = names
plot_df["similarity"] = cos_sim
plot_df["category"] = "Other"
plot_df["opacity"] = 0.5  # Default opacity

# Mark selected company (Red)
plot_df.loc[index, "category"] = "Selected"
plot_df.loc[index, "opacity"] = 1.0

# Mark top similar companies (Green)
if top_indices:
    plot_df.loc[top_indices, "category"] = "Similar"
    plot_df.loc[top_indices, "opacity"] = 0.9

# Define color mapping
color_map = {
    "Selected": "red",
    "Similar": "green", 
    "Other": "lightgrey"
}

# Create the plot
fig = px.scatter(
    plot_df,
    x="x",
    y="y",
    width=800,
    height=600,
    color="category",
    hover_name="name",
    hover_data={"similarity": ":.4f"},
    color_discrete_map=color_map,
    title=f"Selected: {selected_name}",
    labels={"x": "Dimension 1", "y": "Dimension 2"}
)

# Update marker properties
fig.update_traces(
    marker=dict(
        size=2,
        line=dict(width=0.5, color='DarkSlateGrey')
    )
)

# Make selected company larger and more prominent
# update selected marker as +

for i, trace in enumerate(fig.data):
    if trace.name == "Selected":
        trace.update(marker=dict(size=12,symbol = 103,line=dict(width=2, color='black')))
    elif trace.name == "Similar":
        trace.update(marker=dict(size=10,symbol = 205))

# Update layout
fig.update_layout(
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255,255,255,0.8)"
    ),
    height=600
)

# Show the plot
st.plotly_chart(fig, use_container_width=True)
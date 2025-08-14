import pandas as pd
import numpy as np
import streamlit as st
from numpy.linalg import norm
import plotly.express as px
import ast  # for parsing lists from CSV

# ====== Loaders with caching ======
@st.cache_data
def load_manual_dedupe(path="manual_dedupe.csv"):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Names", "Associated Names", "Review"])
        df.to_csv(path, index=False)
        return df

@st.cache_data
def load_main_data():
    df = pd.read_csv("joined_deduped_brokers.csv")
    embeddings = np.load("embeddings_numpy.npy")
    X_embedded = np.load("X_embedded_tsne.npy")
    return df, embeddings, X_embedded

# ====== Data Load ======
man_dedupe = load_manual_dedupe()
df, embeddings, X_embedded = load_main_data()
names = df["Company_Name"].tolist()

# Parse Associated Names column into lists
associated_lists = man_dedupe["Associated Names"].apply(
    lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else []
)

# Check if a name exists in any Associated Names list
def is_name_in_associated(name):
    return any(name in sublist for sublist in associated_lists)

# Mask for already deduped companies
already_in_dedupe = df["Company_Name"].isin(man_dedupe["Names"]) | df["Company_Name"].apply(is_name_in_associated)

# Filter names for selection
names_for_selection = df.loc[~already_in_dedupe, "Company_Name"].tolist()

# ====== Sidebar Controls ======
st.sidebar.header("Controls")
threshold = st.sidebar.slider("Cosine Similarity Threshold", 0.0, 1.0, 0.7, 0.01)
top_n = st.sidebar.selectbox(
    "Number of Top Similar Companies to Display",
    options=[10, 20, 30, 40, 50, 75, 100, 200, 250, 300, 350, 400, 500, 600, 700, 1000, len(names)],
    index=4
)
st.write(len(names_for_selection))
# ====== Select Company ======
selected_name = st.selectbox("Select a name:", names_for_selection)
index = names.index(selected_name)
embed = embeddings[index]

# ====== Cosine Similarity ======
norms = np.linalg.norm(embeddings, axis=1) * norm(embed)
cos_sim = np.dot(embeddings, embed) / norms

# Filter top matches (excluding self and already-deduped)
sorted_indices = np.argsort(-cos_sim)
sorted_indices = sorted_indices[(sorted_indices != index) & (~already_in_dedupe.values[sorted_indices])]
filtered_indices = sorted_indices[cos_sim[sorted_indices] >= threshold][:top_n]

# ====== Layout ======
col1, col2 = st.columns(2)

# ====== Column 1: Similar companies ======
with col1:
    st.subheader(f"Top {len(filtered_indices)} Similar Companies (>{threshold})")
    selected_names_via_checkbox = []
    name_for_review = []

    for i in filtered_indices:
        col_a, col_b = st.columns([1, 4])
        with col_a:
            if st.checkbox("Review", key=f"review_{names[i]}"):
                name_for_review.append(names[i])
        with col_b:
            if st.checkbox(f"**{names[i]}** : {cos_sim[i]:.4f}", key=f"select_{names[i]}", value=True):
                selected_names_via_checkbox.append(names[i])

# ====== Visualization ======
# plot_df = pd.DataFrame(X_embedded, columns=["x", "y"])
# plot_df["name"] = names
# plot_df["similarity"] = cos_sim
# plot_df["category"] = "Other"
# plot_df["opacity"] = 0.5

# # Highlight selected and similar companies
# plot_df.loc[index, ["category", "opacity"]] = ["Selected", 1.0]
# plot_df.loc[filtered_indices, ["category", "opacity"]] = ["Similar", 0.9]

# fig = px.scatter(
#     plot_df,
#     x="x", y="y",
#     color="category",
#     hover_name="name",
#     hover_data={"similarity": ":.4f"},
#     color_discrete_map={"Selected": "red", "Similar": "green", "Other": "lightgrey"},
#     width=800, height=600,
#     title=f"Selected: {selected_name}"
# )

# fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color="DarkSlateGrey")))
# fig.for_each_trace(
#     lambda t: t.update(marker=dict(size=12, symbol=103, line=dict(width=2, color="black"))) if t.name == "Selected" else None
# )
# fig.for_each_trace(
#     lambda t: t.update(marker=dict(size=10, symbol=205)) if t.name == "Similar" else None
# )

# st.plotly_chart(fig, use_container_width=True)

# ====== Column 2: Save results ======
with col2:
    if selected_names_via_checkbox:
        st.write("### Selected Companies:", selected_names_via_checkbox)

    if st.button("Add to Manual Dedupe List"):
        if selected_name in man_dedupe["Names"].values or is_name_in_associated(selected_name):
            st.warning(f"**{selected_name}** is already in the manual dedupe list.")
        else:
            new_data = pd.DataFrame({
                "Names": [selected_name],
                "Associated Names": [selected_names_via_checkbox],
                "Review": [name_for_review]
            })
            new_data.to_csv("manual_dedupe.csv", mode="a", index=False, header=False)
            st.success(f"**{selected_name}** and its associated names have been added.")
        

# ====== Download button ======
with open("manual_dedupe.csv", "rb") as f:
    st.download_button("Download CSV", f, "manual_dedupe.csv", "text/csv")

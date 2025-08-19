import pandas as pd
import numpy as np
import streamlit as st
from numpy.linalg import norm
import plotly.express as px
import ast
from sklearn.metrics.pairwise import cosine_similarity



df_address = pd.read_csv("classified_comnpany_person_in_brokers.csv")
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

@st.cache_data
def precompute_normalized_embeddings(embeddings):
    """Pre-normalize embeddings for faster cosine similarity"""
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

@st.cache_data
def create_dedupe_lookup_set(man_dedupe):
    """Create a set for O(1) lookup of already deduped companies"""
    dedupe_set = set(man_dedupe["Names"].tolist())
    
    # Parse and add all associated names
    for associated_str in man_dedupe["Associated Names"]:
        if pd.notna(associated_str) and isinstance(associated_str, str):
            try:
                associated_list = ast.literal_eval(associated_str)
                dedupe_set.update(associated_list)
            except:
                pass
    
    return dedupe_set

# ====== Data Load ======
man_dedupe = load_manual_dedupe()
df, embeddings, X_embedded = load_main_data()

# Pre-compute normalized embeddings and dedupe lookup
normalized_embeddings = precompute_normalized_embeddings(embeddings)
dedupe_lookup = create_dedupe_lookup_set(man_dedupe)

# Create filtered names list once
names = df["Company_Name"].tolist()
already_in_dedupe = df["Company_Name"].isin(dedupe_lookup)
names_for_selection = df.loc[~already_in_dedupe, "Company_Name"].tolist()



# ====== Sidebar Controls ======
st.sidebar.header("Controls")
threshold = st.sidebar.slider("Cosine Similarity Threshold", 0.0, 1.0, 0.7, 0.01)
top_n = st.sidebar.selectbox(
    "Number of Top Similar Companies to Display",
    options=[10, 20, 30, 40, 50, 75, 100, 200, 250, 300, 350, 400, 500],
    index=4
)

st.write(f"Already processed: {len(names) - len(names_for_selection)} companies")

# ====== Select Company ======
if not names_for_selection:
    st.warning("All companies have been processed!")
    st.stop()

selected_name = st.selectbox("Select a name:", names_for_selection)

# ====== Optimized Similarity Calculation ======
@st.cache_data
def calculate_similarities(selected_name, _normalized_embeddings, _names, _dedupe_lookup, threshold, top_n):
    """Calculate similarities with caching"""
    try:
        index = _names.index(selected_name)
    except ValueError:
        return [], []
    
    # Fast cosine similarity using pre-normalized embeddings
    selected_embedding = _normalized_embeddings[index:index+1]  # Keep 2D shape
    cos_sim = np.dot(_normalized_embeddings, selected_embedding.T).flatten()
    
    # Create boolean mask for valid candidates
    valid_mask = np.ones(len(_names), dtype=bool)
    # valid_mask[index] = False  # Exclude self
    
    # Exclude already deduped companies
    for i, name in enumerate(_names):
        if name in _dedupe_lookup:
            valid_mask[i] = False
    
    # Apply threshold filter
    threshold_mask = cos_sim >= threshold
    final_mask = valid_mask & threshold_mask
    
    # Get top matches
    valid_indices = np.where(final_mask)[0]
    valid_similarities = cos_sim[valid_indices]
    
    # Sort by similarity (descending)
    sorted_order = np.argsort(-valid_similarities)[:top_n]
    final_indices = valid_indices[sorted_order]
    final_similarities = valid_similarities[sorted_order]
    
    return final_indices.tolist(), final_similarities.tolist()

# Calculate similarities
filtered_indices, similarities = calculate_similarities(
    selected_name, normalized_embeddings, names, dedupe_lookup, threshold, top_n
)

# ====== Layout ======
col1, col2 = st.columns(2)

# ====== Column 1: Similar companies ======
with col1:
    st.subheader(f"Top {len(filtered_indices)} Similar Companies (>{threshold})")
    
    if not filtered_indices:
        st.info("No similar companies found above the threshold.")
    
    # Use session state for selections to improve performance
    if 'selected_companies' not in st.session_state:
        st.session_state.selected_companies = []
    if 'review_companies' not in st.session_state:
        st.session_state.review_companies = []
    
    # Create form for batch processing
    with st.form("similarity_form"):
        selected_names_via_checkbox = []
        name_for_review = []
        
        for idx, (i, similarity) in enumerate(zip(filtered_indices, similarities)):
            company_name = names[i]
            col_a, col_b = st.columns([1, 4])
            
            with col_a:
                if st.checkbox("Review", key=f"review_{idx}"):
                    name_for_review.append(company_name)
            
            with col_b:
                address = df_address[df_address["COMPANY_NAME"] == company_name]["STATE"].values
                # print(company_name, address)
                if st.checkbox(f"**{company_name}** : {similarity:.4f} Address :{address[0]}", key=f"select_{idx}", value=True):
                    selected_names_via_checkbox.append(company_name)
        
        # Submit button for the form
        form_submitted = st.form_submit_button("Update Selections")

# ====== Column 2: Save results ======
with col2:
    if selected_names_via_checkbox:
        st.write("### Selected Companies:")
        for name in selected_names_via_checkbox:
            st.write(f"- {name}")

    if st.button("Add to Manual Dedupe List", type="primary"):
        if selected_name in dedupe_lookup:
            st.warning(f"**{selected_name}** is already in the manual dedupe list.")
        else:
            new_data = pd.DataFrame({
                "Names": [selected_name],
                "Associated Names": [selected_names_via_checkbox],
                "Review": [name_for_review]
            })
            new_data.to_csv("manual_dedupe.csv", mode="a", index=False, header=False)
            st.success(f"**{selected_name}** and its associated names have been added.")
            
            # Clear cache to refresh data
            st.cache_data.clear()
            st.rerun()

# ====== Performance Info ======
with st.expander("Performance Info"):
    st.write(f"Total companies: {len(names)}")
    st.write(f"Already processed: {len(names) - len(names_for_selection)}")
    st.write(f"Remaining: {len(names_for_selection)}")
    st.write(f"Embedding dimensions: {embeddings.shape}")

# ====== Download button ======
try:
    with open("manual_dedupe.csv", "rb") as f:
        st.download_button("Download CSV", f, "manual_dedupe.csv", "text/csv")
except FileNotFoundError:
    st.info("No manual dedupe file found yet.")

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

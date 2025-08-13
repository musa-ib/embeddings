import pandas as pd
import numpy as np
import streamlit as st
from numpy.linalg import norm
import plotly.express as px



try:
    man_dedupe = pd.read_csv('manual_dedupe.csv')
except FileNotFoundError:
    pd.DataFrame(columns=['Names', 'Associated Names','Review']).to_csv('manual_dedupe.csv', index=False)


man_dedupe = pd.read_csv('manual_dedupe.csv')
def find_in_associated_names(name):
    for i in man_dedupe['Associated Names']:
        if name in i:
            return True
    return False

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
top_n = st.sidebar.selectbox("Number of Top Similar Companies to Display", options=[10, 20, 30, 40, 50,75,100,200,250,300,350,400,500,len(names)], index=4)



formatted_options = []
for name in names:
    if name in man_dedupe["Names"].values or find_in_associated_names(name):
        formatted_options.append(f" {name}  🟢")  # Green circle prefix for highlighted names
    else:
        formatted_options.append(name)

# Create a mapping to get original name from formatted option
option_to_name = {}
for i, name in enumerate(names):
    option_to_name[formatted_options[i]] = name

# Updated selectbox with formatted options
selected_option = st.selectbox("Select a name:", formatted_options)
selected_name = option_to_name[selected_option]


# selected_name = st.selectbox("Select a name:", names)
index = names.index(selected_name)
embed = embeddings[index]

# Compute cosine similarities
cos_sim = np.array([np.dot(i, embed) / (norm(i) * norm(embed)) for i in embeddings])

# Sort and filter
sorted_indices = np.argsort(-cos_sim)
sorted_indices = sorted_indices[sorted_indices != index]  # Remove selected company
filtered_indices = [i for i in sorted_indices if cos_sim[i] >= threshold]
top_indices = filtered_indices[:top_n]

col1 , col2 = st.columns(2)




# Display results and add checkboxes
with col1:
    st.subheader(f"Top {len(top_indices)} Similar Companies with Similarity > {threshold}")

    selected_names_via_checkbox=[]
    name_for_review = []


    if top_indices:
        for i in top_indices:
            col_a, col_b = st.columns([1, 4])
            
            with col_a:
                # Second checkbox (you can customize the label/purpose)
                second_check = st.checkbox("Review", key=f"second_{names[i]}", value=False)
                if second_check:
                    name_for_review.append(names[i])
            
            with col_b:
                # Original checkbox
                if st.checkbox(f"**{names[i]}** : {cos_sim[i]:.4f}", key=names[i], value=True):
                    selected_names_via_checkbox.append(names[i])


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
    title=f"Selected: {selected_name}"
    # labels={"x": "Dimension 1", "y": "Dimension 2"}
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

# ==  Find Selected name in Associated Names ==


with col2:
    if selected_names_via_checkbox:
        st.write("### Selected Companies:" , selected_names_via_checkbox)


    if st.button("Add to Manual Dedupe List"):
        if not selected_names_via_checkbox:
            st.warning("Please select at least one company to add.")
        else:
            if selected_name in man_dedupe['Names'].values or find_in_associated_names(selected_name):
                st.write(f"**{selected_name}** is already in the manual dedupe list.")
            else:
                new_data_point = pd.DataFrame({'Names': [selected_name], 'Associated Names': [selected_names_via_checkbox],'Review': [name_for_review]})
                new_data_point.to_csv('manual_dedupe.csv', mode='a', index=False, header=False)
                
                st.success(f"**{selected_name}** and its associated names have been added to the manual dedupe list.")

csv_file_path = 'manual_dedupe.csv'
with open(csv_file_path, 'rb') as f:
    st.download_button(
        label="Download CSV", 
        data=f, 
        file_name="manual_dedupe.csv", 
        mime="text/csv"
    )
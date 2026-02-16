import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
import os
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Recommender Evaluation Lab — App Algorithm vs Baseline (with neighbor filtering)")

# ---------- LOAD DATA ----------
@st.cache_data
def load_data(path="ratings.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path} in {os.getcwd()}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty")
    return df

ratings = load_data()

# ---------- BUILD USER-MOVIE MATRICES ----------
user_movie_matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating")
user_means = user_movie_matrix.mean(axis=1)
mean_centered = user_movie_matrix.sub(user_means, axis=0).fillna(0)

all_users = user_movie_matrix.index.tolist()

# ---------- SIDEBAR CONTROLS ----------
st.sidebar.header("Evaluation Parameters")
num_test_users = st.sidebar.slider("Number of test users", 5, 100, 20, step=5)
num_known_ratings = st.sidebar.slider("Known ratings per user (train)", 1, 50, 10)
num_neighbors = st.sidebar.slider("Number of cosine neighbours", 1, 50, 5)
min_overlap = st.sidebar.slider("Minimum overlap (shared movies)", 1, 20, 5)
min_neighbor_ratings = st.sidebar.slider("Minimum ratings for neighbor users", 1, 50, 10)
random_seed = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.write("Notes:")
st.sidebar.write("- App algorithm = mean-centered, weighted-by-similarity, overlap-filtered.")
st.sidebar.write("- Baseline = simple mean of neighbors' ratings for the item.")
st.sidebar.write("- Neighbor users must have at least N ratings (min_neighbor_ratings).")

# ---------- UTILS ----------
def rmse_from_lists(actuals, preds):
    if len(actuals) == 0:
        return None
    return float(np.sqrt(np.mean((np.array(actuals) - np.array(preds)) ** 2)))

# ---------- RUN EVALUATION ----------
if st.button("Run Evaluation"):
    random.seed(random_seed)
    np.random.seed(random_seed)

    candidate_users = [u for u in all_users if user_movie_matrix.loc[u].dropna().shape[0] > num_known_ratings]
    if len(candidate_users) < num_test_users:
        st.warning(f"Only {len(candidate_users)} users have more than {num_known_ratings} ratings. Reducing test users.")
    test_users = random.sample(candidate_users, min(num_test_users, len(candidate_users)))

    rmses_app = []
    rmses_base = []
    per_user_results = []

    progress = st.progress(0)
    for i, uid in enumerate(test_users):
        ur = user_movie_matrix.loc[uid].dropna()
        ur_idx = list(ur.index)
        random.shuffle(ur_idx)
        train_ids = ur_idx[:num_known_ratings]
        test_ids = ur_idx[num_known_ratings:]

        # build target vector
        user_vector = pd.Series(0, index=mean_centered.columns, dtype=float)
        for m in train_ids:
            user_vector[m] = user_movie_matrix.loc[uid, m]

        train_values = [user_vector[m] for m in train_ids if user_vector[m] != 0]
        if len(train_values) == 0:
            continue
        target_mean = float(np.mean(train_values))
        user_vector_centered = (user_vector - target_mean).fillna(0)

        sims = cosine_similarity([user_vector_centered], mean_centered.values)[0]

        # ----------- FILTER NEIGHBORS -----------
        train_mask = (user_vector_centered != 0).astype(int)
        overlaps = (mean_centered != 0).dot(train_mask)
        valid_user_indices = np.where(overlaps >= min_overlap)[0]

        if len(valid_user_indices) > 0:
            neighbor_counts = user_movie_matrix.iloc[valid_user_indices].count(axis=1).to_numpy()
            mask = neighbor_counts >= min_neighbor_ratings
            valid_user_indices = valid_user_indices[mask]
        else:
            valid_user_indices = np.array([], dtype=int)

        if valid_user_indices.size == 0:
            progress.progress((i + 1)/len(test_users))
            continue

        valid_sims = sims[valid_user_indices]
        sorted_idx = np.argsort(valid_sims)
        take = min(num_neighbors, len(valid_user_indices))
        top_valid_indices = valid_user_indices[sorted_idx[-take:]]
        top_sims = sims[top_valid_indices]
        top_user_ids = list(user_movie_matrix.index[top_valid_indices])

        preds_app = []
        actuals_app = []
        preds_base = []
        actuals_base = []

        for m in test_ids:
            true_rating = user_movie_matrix.loc[uid, m]

            neigh_ratings_df = ratings[(ratings["userId"].isin(top_user_ids)) & (ratings["movieId"] == m)][["userId","rating"]]
            if neigh_ratings_df.empty:
                continue

            # Baseline
            base_pred = float(neigh_ratings_df["rating"].mean())
            preds_base.append(base_pred)
            actuals_base.append(true_rating)

            # App algorithm
            num = 0.0
            den = 0.0
            for uidx, sim_val in zip(top_valid_indices, top_sims):
                neighbor_id = user_movie_matrix.index[uidx]
                neigh_centered = mean_centered.loc[neighbor_id, m] if m in mean_centered.columns else 0
                if neigh_centered != 0 and sim_val > 0:
                    num += sim_val * neigh_centered
                    den += abs(sim_val)
            if den > 0:
                pred_app = target_mean + (num/den)
                preds_app.append(pred_app)
                actuals_app.append(true_rating)
            else:
                preds_app.append(base_pred)
                actuals_app.append(true_rating)

        rmse_a = rmse_from_lists(actuals_app, preds_app)
        rmse_b = rmse_from_lists(actuals_base, preds_base)
        if rmse_a is not None:
            rmses_app.append(rmse_a)
        if rmse_b is not None:
            rmses_base.append(rmse_b)

        per_user_results.append((uid, rmse_a, rmse_b, len(actuals_app)))
        progress.progress((i + 1)/len(test_users))

    # ---------- NUMERIC RESULTS ----------
    st.write("### Numeric Results")
    if rmses_app:
        st.write(f"App algorithm — mean RMSE: **{np.mean(rmses_app):.4f}**, median RMSE: **{np.median(rmses_app):.4f}**, users evaluated: {len(rmses_app)}")
    else:
        st.write("App algorithm — no RMSE values computed.")
    if rmses_base:
        st.write(f"Baseline — mean RMSE: **{np.mean(rmses_base):.4f}**, median RMSE: **{np.median(rmses_base):.4f}**, users evaluated: {len(rmses_base)}")
    else:
        st.write("Baseline — no RMSE values computed.")

    # ---------- BOX PLOT ----------
    st.write("### RMSE Distribution: App vs Baseline")
    fig, ax = plt.subplots(figsize=(8,5))
    data_to_plot = []
    labels = []
    if rmses_app:
        data_to_plot.append(rmses_app)
        labels.append("App")
    if rmses_base:
        data_to_plot.append(rmses_base)
        labels.append("Baseline")
    if data_to_plot:
        ax.boxplot(data_to_plot, labels=labels, showmeans=True)
        ax.set_ylabel("RMSE")
        ax.set_title("RMSE Distribution per-user")
        st.pyplot(fig)

    # ---------- PER-USER TABLE ----------
    st.write("### Sample Per-user RMSE")
    if per_user_results:
        df_per_user = pd.DataFrame(per_user_results, columns=["userId","rmse_app","rmse_base","n_predictions"])
        st.dataframe(df_per_user.head(50))
    else:
        st.write("No per-user results.")

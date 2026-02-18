import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
import os
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Recommender Evaluation Lab — App Algorithm vs Baseline vs NMF")

# ---------- CONFIG ----------
NPZ_MODEL_PATH = "nmf_300f_top3000.npz"   # <-- HARD CODED

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

# ---------- BUILD MATRICES ----------
user_movie_matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating")
user_means = user_movie_matrix.mean(axis=1)
mean_centered = user_movie_matrix.sub(user_means, axis=0).fillna(0)
all_users = user_movie_matrix.index.tolist()

# ---------- SIDEBAR ----------
st.sidebar.header("Evaluation Parameters")
num_test_users = st.sidebar.slider("Number of test users", 5, 100, 20, step=5)
num_known_ratings = st.sidebar.slider("Known ratings per user", 1, 50, 10)
num_neighbors = st.sidebar.slider("Cosine neighbours", 1, 50, 5)
min_overlap = st.sidebar.slider("Minimum overlap", 1, 20, 5)
min_neighbor_ratings = st.sidebar.slider("Min neighbor ratings", 1, 50, 10)
random_seed = st.sidebar.number_input("Random seed", value=42, step=1)

# ---------- UTILS ----------
def rmse_from_lists(actuals, preds):
    if len(actuals) == 0:
        return None
    return float(np.sqrt(np.mean((np.array(actuals) - np.array(preds)) ** 2)))

# ---------- LOAD NPZ ----------
def load_npz_model(path):
    if not os.path.exists(path):
        st.error(f"NMF model not found: {path}")
        st.stop()
    d = np.load(path, allow_pickle=True)
    return d["H"], d["movie_ids"].astype(int)

H, nmf_movie_ids = load_npz_model(NPZ_MODEL_PATH)
H_pinv = np.linalg.pinv(H)

# ---------- RUN ----------
if st.button("Run Evaluation"):
    random.seed(random_seed)
    np.random.seed(random_seed)

    candidate_users = [u for u in all_users if user_movie_matrix.loc[u].dropna().shape[0] > num_known_ratings]
    test_users = random.sample(candidate_users, min(num_test_users, len(candidate_users)))

    rmses_app, rmses_base, rmses_nmf = [], [], []
    per_user_results = []

    progress = st.progress(0)

    for i, uid in enumerate(test_users):
        ur = user_movie_matrix.loc[uid].dropna()
        ur_idx = list(ur.index)
        random.shuffle(ur_idx)

        train_ids = ur_idx[:num_known_ratings]
        test_ids = ur_idx[num_known_ratings:]
        if not test_ids:
            continue

        # ---------- TARGET VECTOR ----------
        user_vector = pd.Series(0, index=mean_centered.columns, dtype=float)
        for m in train_ids:
            user_vector[m] = user_movie_matrix.loc[uid, m]

        train_vals = [user_vector[m] for m in train_ids if user_vector[m] != 0]
        if not train_vals:
            continue

        target_mean = float(np.mean(train_vals))
        user_vector_centered = (user_vector - target_mean).fillna(0)
        sims = cosine_similarity([user_vector_centered], mean_centered.values)[0]

        # ---------- FILTER NEIGHBORS ----------
        train_mask = (user_vector_centered != 0).astype(int)
        overlaps = (mean_centered != 0).dot(train_mask)
        valid_user_indices = np.where(overlaps >= min_overlap)[0]

        if valid_user_indices.size == 0:
            continue

        valid_sims = sims[valid_user_indices]
        sorted_idx = np.argsort(valid_sims)
        take = min(num_neighbors, len(valid_user_indices))
        top_valid_indices = valid_user_indices[sorted_idx[-take:]]
        top_sims = sims[top_valid_indices]
        top_user_ids = list(user_movie_matrix.index[top_valid_indices])

        preds_app, actuals_app = [], []
        preds_base, actuals_base = [], []
        preds_nmf, actuals_nmf = [], []

        # ---------- NMF USER LATENT ----------
        user_vec_nmf = pd.Series(0, index=nmf_movie_ids)
        for tid in train_ids:
            if tid in user_vec_nmf.index:
                user_vec_nmf[tid] = user_movie_matrix.loc[uid, tid]
        mean_val = user_vec_nmf[user_vec_nmf > 0].mean() if (user_vec_nmf > 0).any() else 0
        centered = (user_vec_nmf - mean_val).fillna(0)
        latent = centered.values @ H_pinv
        nmf_preds_full = latent @ H + mean_val
        nmf_dict = dict(zip(nmf_movie_ids, nmf_preds_full))

        for m in test_ids:
            true_rating = user_movie_matrix.loc[uid, m]

            neigh_df = ratings[
                (ratings["userId"].isin(top_user_ids)) &
                (ratings["movieId"] == m)
            ]

            if neigh_df.empty:
                continue

            # BASELINE
            base_pred = float(neigh_df["rating"].mean())
            preds_base.append(base_pred)
            actuals_base.append(true_rating)

            # APP
            num, den = 0.0, 0.0
            for uidx, sim_val in zip(top_valid_indices, top_sims):
                neighbor_id = user_movie_matrix.index[uidx]
                neigh_centered = mean_centered.loc[neighbor_id, m]
                if neigh_centered != 0 and sim_val > 0:
                    num += sim_val * neigh_centered
                    den += abs(sim_val)
            pred_app = target_mean + (num / den) if den > 0 else base_pred
            preds_app.append(pred_app)
            actuals_app.append(true_rating)

            # NMF
            pred_nmf = nmf_dict.get(m, base_pred)
            preds_nmf.append(pred_nmf)
            actuals_nmf.append(true_rating)

        rmse_a = rmse_from_lists(actuals_app, preds_app)
        rmse_b = rmse_from_lists(actuals_base, preds_base)
        rmse_n = rmse_from_lists(actuals_nmf, preds_nmf)

        if rmse_a: rmses_app.append(rmse_a)
        if rmse_b: rmses_base.append(rmse_b)
        if rmse_n: rmses_nmf.append(rmse_n)

        per_user_results.append((uid, rmse_a, rmse_b, rmse_n, len(actuals_app)))
        progress.progress((i + 1) / len(test_users))

    # ---------- RESULTS ----------
    st.write("### Numeric Results")
    if rmses_app: st.write(f"App RMSE mean: {np.mean(rmses_app):.4f}")
    if rmses_base: st.write(f"Baseline RMSE mean: {np.mean(rmses_base):.4f}")
    if rmses_nmf: st.write(f"NMF RMSE mean: {np.mean(rmses_nmf):.4f}")

    fig, ax = plt.subplots()
    ax.boxplot([rmses_app, rmses_base, rmses_nmf], labels=["App","Baseline","NMF"], showmeans=True)
    st.pyplot(fig)

    df = pd.DataFrame(per_user_results, columns=["userId","rmse_app","rmse_base","rmse_nmf","n_preds"])
    st.dataframe(df.head(50))

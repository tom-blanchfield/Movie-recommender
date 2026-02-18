import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
import random, os
import matplotlib.pyplot as plt

st.title("Evaluation Lab")

ratings = pd.read_csv("ratings.csv")
user_movie_matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating")

# ---------- SIDEBAR ----------
nmf_factors = st.sidebar.slider("Latent Factors",2,300,20)
use_pretrained = st.sidebar.checkbox("Use Pretrained NPZ")
npz_path = st.sidebar.text_input("NPZ Path","nmf_300f_top3000.npz")

# ---------- PRETRAIN LOAD ----------
@st.cache_data
def load_npz(path):
    d=np.load(path,allow_pickle=True)
    return d["H"], d["movie_ids"].astype(int)

# ---------- LIVE FIT ----------
@st.cache_data
def fit_live(mat,f):
    m=mat.fillna(mat.mean())
    model=NMF(n_components=f,max_iter=500)
    W=model.fit_transform(m)
    H=model.components_
    return pd.DataFrame(W@H,index=mat.index,columns=mat.columns)

if use_pretrained:
    H, nmf_movie_ids = load_npz(npz_path)
    H_pinv = np.linalg.pinv(H)
else:
    nmf_matrix = fit_live(user_movie_matrix,nmf_factors)

# ---------- RUN ----------
if st.button("Run"):
    rmses=[]
    users=random.sample(list(user_movie_matrix.index),20)

    for uid in users:
        row=user_movie_matrix.loc[uid].dropna()
        if len(row)<5: continue
        m=row.index[0]
        true=row.iloc[0]

        if use_pretrained:
            user_vec=pd.Series(0,index=nmf_movie_ids)
            if m in user_vec.index: user_vec[m]=true
            mean=user_vec[user_vec>0].mean() if (user_vec>0).any() else 0
            centered=(user_vec-mean).fillna(0)
            latent=centered.values@H_pinv
            preds=latent@H+mean
            pred=dict(zip(nmf_movie_ids,preds)).get(m)
        else:
            pred=nmf_matrix.loc[uid,m]

        if pred is not None:
            rmses.append((true-pred)**2)

    rmse=np.sqrt(np.mean(rmses))
    st.write("RMSE:",rmse)

# =============================================================================
# HYBRID MOVIE RECOMMENDATION ENGINE — Streamlit App
# =============================================================================
#
# SETUP INSTRUCTIONS
# ------------------
# 1. Install dependencies:
#       pip install streamlit pandas joblib scikit-surprise requests numpy
#
# 2. Get a FREE OMDb API key:
#       a. Visit https://www.omdbapi.com/apikey.aspx
#       b. Choose the FREE tier (1,000 requests/day)
#       c. Verify your email — you'll receive the key instantly
#       d. Paste the key below in the OMDB_API_KEY constant
#
# 3. Place these files in the SAME directory as app.py:
#       - best_svd_model.joblib
#       - movies_df.pkl
#
# 4. Run:
#       streamlit run steamlit_app.py
#
# =============================================================================

import html
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import requests
import os
import numpy as np
from surprise import SVD

# ── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ──
#  CONFIGURATION  (edit here)
# ── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ──

OMDB_API_KEY = "85975d64"          # ← paste your free key here
OMDB_BASE_URL = "https://www.omdbapi.com/"

PLACEHOLDER_POSTER = (
    "https://via.placeholder.com/300x450/1a1a2e/e94560?"
    "text=No+Poster+Available"
)

MODEL_PATH  = "best_svd_model.joblib"
MOVIES_PATH = "movies_df.pkl"

RATING_SCALE_MIN = 0.5
RATING_SCALE_MAX = 5.0

# ── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ──
#  PAGE CONFIG  (must be first Streamlit call)
# ── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ──

st.set_page_config(
    page_title="CineMatch — Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ──
#  CUSTOM CSS  (dark-mode cinema theme)
# ── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ──

st.markdown("""
<style>
/* ── Base dark background ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d0d1a;
    color: #e0e0e0;
}
[data-testid="stSidebar"] {
    background-color: #12122a;
    border-right: 1px solid #2a2a4a;
}
[data-testid="stSidebar"] * { color: #d0d0f0 !important; }

/* ── Header ── */
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #e94560 0%, #f5a623 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.hero-sub {
    color: #8888aa;
    font-size: 1rem;
    margin-top: 0;
    margin-bottom: 1.5rem;
}

/* ── Section label ── */
.section-label {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e94560;
    margin-bottom: 1rem;
}
.info-box {
    background: #1a1a2e;
    border-left: 4px solid #e94560;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 0.88rem;
    color: #aaaacc;
    margin-bottom: 1.2rem;
}
</style>
""", unsafe_allow_html=True)


# ── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ──
#  DATA & MODEL LOADERS  (cached)
# ── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ──

@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_movies():
    return pd.read_pickle(MOVIES_PATH)


@st.cache_data(show_spinner=False)
def get_all_genres(_movies_df: pd.DataFrame) -> list[str]:
    genres = set()
    for g_string in _movies_df["genres"].dropna():
        for g in g_string.split("|"):
            if g and g.lower() != "(no genres listed)":
                genres.add(g)
    return sorted(genres)


# ── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ──
#  OMDb FETCHER  (cached per title)
# ── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ──

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_movie_details(title: str) -> dict:
    default = {
        "poster_url": PLACEHOLDER_POSTER,
        "description": "Plot information unavailable.",
        "year": "N/A",
        "director": "N/A",
        "imdb_rating": "N/A",
    }

    if not OMDB_API_KEY or OMDB_API_KEY == "YOUR_OMDB_KEY_HERE":
        return default

    clean_title = title.strip()
    if clean_title.endswith(")"):
        last_open = clean_title.rfind("(")
        if last_open != -1:
            candidate = clean_title[last_open + 1:-1].strip()
            if candidate.isdigit() and len(candidate) == 4:
                clean_title = clean_title[:last_open].strip()

    if clean_title.endswith(")"):
        last_open = clean_title.rfind("(")
        if last_open != -1:
            inner = clean_title[last_open + 1:-1].strip()
            if inner.lower().startswith("a.k.a."):
                aka = inner[6:].strip()
                if aka:
                    clean_title = aka

    if clean_title.endswith(")"):
        last_open = clean_title.rfind("(")
        if last_open != -1:
            clean_title = clean_title[:last_open].strip()

    for suffix in [", The", ", A", ", An"]:
        if clean_title.endswith(suffix):
            clean_title = suffix[2:] + " " + clean_title[:-len(suffix)]
            break

    try:
        resp = requests.get(
            OMDB_BASE_URL,
            params={"t": clean_title, "apikey": OMDB_API_KEY, "plot": "short"},
            timeout=5,
        )
        data = resp.json()
    except Exception:
        return default

    if data.get("Response") != "True":
        return default

    poster = data.get("Poster", "")
    return {
        "poster_url": poster if (poster and poster != "N/A") else PLACEHOLDER_POSTER,
        "description": data.get("Plot", "Plot unavailable."),
        "year": data.get("Year", "N/A"),
        "director": data.get("Director", "N/A"),
        "imdb_rating": data.get("imdbRating", "N/A"),
    }


# ── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ──
#  SIMILARITY HELPERS
# ── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ──

def compute_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))


# ── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ──
#  RECOMMENDATION ENGINES
# ── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ──

def recommend_for_existing_user(
    model: SVD,
    movies_df: pd.DataFrame,
    user_id: int,
    top_n: int,
) -> pd.DataFrame:
    """Route A — Collaborative Filtering via the trained SVD model."""
    trainset = model.trainset
    try:
        inner_uid = trainset.to_inner_uid(user_id)
    except ValueError:
        return pd.DataFrame()

    rated_inner = {iid for (iid, _) in trainset.ur[inner_uid]}

    scores = []
    for inner_iid in trainset.all_items():
        if inner_iid in rated_inner:
            continue
        raw_iid = trainset.to_raw_iid(inner_iid)
        pred = model.predict(user_id, raw_iid)
        scores.append((raw_iid, round(pred.est, 3)))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_scores = scores[:top_n]

    if not top_scores:
        return pd.DataFrame()

    score_map = {mid: sc for mid, sc in top_scores}
    result = movies_df[movies_df["movieId"].isin(score_map.keys())].copy()
    result["predicted_score"] = result["movieId"].map(score_map)
    result.sort_values("predicted_score", ascending=False, inplace=True)
    return result.reset_index(drop=True)


def recommend_via_similar_users(
    model: SVD,
    movies_df: pd.DataFrame,
    user_id: int,
    top_n: int,
) -> pd.DataFrame:
    """Finds users with similar SVD latent factors and returns what they highly rated."""
    trainset = model.trainset
    try:
        target_inner_uid = trainset.to_inner_uid(user_id)
    except ValueError:
        return pd.DataFrame()

    # Get user matrix vectors (pu)
    user_vectors = model.pu
    target_vector = user_vectors[target_inner_uid]

    # Calculate similarity across all users
    similarities = []
    for inner_uid in range(trainset.n_users):
        if inner_uid == target_inner_uid:
            continue
        sim = compute_cosine_similarity(target_vector, user_vectors[inner_uid])
        similarities.append((inner_uid, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_similar_inner_uids = [x[0] for x in similarities[:10]]  # Top 10 neighbors

    # Collate candidate movies highly liked by these neighbors
    candidate_scores = {}
    rated_by_target = {iid for (iid, _) in trainset.ur[target_inner_uid]}

    for inner_uid in top_similar_inner_uids:
        for inner_iid, rating in trainset.ur[inner_uid]:
            if inner_iid in rated_by_target:
                continue
            # Weight candidate by how much the neighbor liked it
            if rating >= 4.0:
                candidate_scores[inner_iid] = candidate_scores.get(inner_iid, 0) + rating

    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    if not sorted_candidates:
        return pd.DataFrame()

    raw_movie_ids = [trainset.to_raw_iid(x[0]) for x in sorted_candidates]
    result = movies_df[movies_df["movieId"].isin(raw_movie_ids)].copy()
    result["predicted_score"] = None  # Neighborhood match indicator
    return result.reset_index(drop=True)


def recommend_similar_items(
    model: SVD,
    movies_df: pd.DataFrame,
    user_id: int,
    top_n: int,
) -> tuple[pd.DataFrame, str]:
    """Finds items similar to the user's favorite movie via SVD item latent vectors (qi)."""
    trainset = model.trainset
    try:
        inner_uid = trainset.to_inner_uid(user_id)
    except ValueError:
        return pd.DataFrame(), ""

    user_history = trainset.ur[inner_uid]
    if not user_history:
        return pd.DataFrame(), ""

    # Locate the highest rated movie in history
    best_inner_iid, max_rating = max(user_history, key=lambda x: x[1])
    fav_movie_raw_id = trainset.to_raw_iid(best_inner_iid)
    
    fav_title_row = movies_df[movies_df["movieId"] == fav_movie_raw_id]
    fav_title = fav_title_row["title"].values[0] if not fav_title_row.empty else "a highly rated movie"

    # Compute item similarities against the item matrix vectors (qi)
    item_vectors = model.qi
    target_item_vector = item_vectors[best_inner_iid]

    item_similarities = []
    rated_inner_ids = {iid for (iid, _) in user_history}

    for inner_iid in range(trainset.n_items):
        if inner_iid in rated_inner_ids:
            continue
        sim = compute_cosine_similarity(target_item_vector, item_vectors[inner_iid])
        item_similarities.append((inner_iid, sim))

    item_similarities.sort(key=lambda x: x[1], reverse=True)
    top_items = item_similarities[:top_n]

    if not top_items:
        return pd.DataFrame(), fav_title

    raw_movie_ids = [trainset.to_raw_iid(x[0]) for x in top_items]
    result = movies_df[movies_df["movieId"].isin(raw_movie_ids)].copy()
    result["predicted_score"] = None
    return result.reset_index(drop=True), fav_title


def recommend_cold_start(
    movies_df: pd.DataFrame,
    selected_genres: list[str],
    top_n: int,
    match_mode: str = "any",
) -> pd.DataFrame:
    """Route B — Content-Based Filtering for new users (cold start)."""
    if not selected_genres:
        return pd.DataFrame()

    def matches(genre_string: str) -> bool:
        if pd.isna(genre_string):
            return False
        movie_genres = set(genre_string.split("|"))
        if match_mode == "all":
            return all(g in movie_genres for g in selected_genres)
        return any(g in movie_genres for g in selected_genres)

    filtered = movies_df[movies_df["genres"].apply(matches)].copy()
    filtered["genre_count"] = filtered["genres"].apply(
        lambda g: len(g.split("|")) if isinstance(g, str) else 0
    )
    filtered = filtered.sample(frac=1, random_state=42).head(top_n * 3)
    filtered.sort_values("genre_count", ascending=False, inplace=True)
    filtered["predicted_score"] = None
    return filtered.head(top_n).reset_index(drop=True)


# ── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ──
#  CARD RENDERER
# ── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ──

def render_movie_card(row: pd.Series, details: dict, custom_badge: str = None) -> None:
    genres_html = "".join(
        f'<span class="tag tag-genre">{html.escape(g)}</span>'
        for g in row["genres"].split("|")
        if g and g.lower() != "(no genres listed)"
    ) if isinstance(row.get("genres"), str) else ""

    score = row.get("predicted_score")
    if score:
        score_html = f'<span class="tag tag-score">&#11088; {score}/5</span>'
    elif custom_badge:
        score_html = f'<span class="tag tag-score">{custom_badge}</span>'
    else:
        score_html = '<span class="tag tag-score">&#127919; Content Match</span>'

    year_html = (
        f'<span class="tag tag-year">&#128197; {html.escape(str(details["year"]))}</span>'
        if details["year"] != "N/A" else ""
    )

    imdb_html = (
        f'<span class="tag tag-rating">IMDb {html.escape(str(details["imdb_rating"]))}</span>'
        if details["imdb_rating"] != "N/A" else ""
    )

    director_html = (
        f'<p class="director-text">&#127916; <b>Dir:</b> {html.escape(str(details["director"]))}</p>'
        if details["director"] != "N/A" else ""
    )

    safe_title       = html.escape(str(row["title"]))
    safe_description = html.escape(str(details["description"]))
    safe_poster      = details["poster_url"]
    safe_placeholder = html.escape(PLACEHOLDER_POSTER, quote=True)

    card_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
      body {{ margin: 0; padding: 0; background: transparent; font-family: sans-serif; }}
      .movie-card {{
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid #2a2a4a;
        border-radius: 14px;
        padding: 14px;
        box-sizing: border-box;
      }}
      .movie-poster {{
        width: 100%; border-radius: 10px;
        object-fit: cover; height: 260px;
        display: block; margin-bottom: 12px;
        background: #0d0d1a;
      }}
      .movie-title {{
        font-size: 0.95rem; font-weight: 700;
        color: #ffffff; margin-bottom: 6px; line-height: 1.3;
      }}
      .meta-row {{ display: flex; gap: 6px; flex-wrap: wrap; margin: 6px 0; }}
      .tag {{
        border-radius: 20px; padding: 2px 10px;
        font-size: 0.72rem; font-weight: 500;
      }}
      .tag-score  {{ background: linear-gradient(90deg,#e94560,#c0392b); color:#fff; }}
      .tag-year   {{ background: linear-gradient(90deg,#f5a623,#c47d0a); color:#fff; }}
      .tag-rating {{ background: linear-gradient(90deg,#27ae60,#1e8449); color:#fff; }}
      .tag-genre  {{ background:#1e2a4a; color:#7fa8ff; }}
      .divider {{ border:none; border-top:1px solid #2a2a4a; margin:8px 0; }}
      .plot-text {{
        font-size:0.78rem; color:#9999bb; line-height:1.5;
        max-height:90px; overflow-y:auto;
      }}
      .director-text {{ font-size:0.72rem; color:#7777aa; margin-top:4px; }}
    </style>
    </head>
    <body>
    <div class="movie-card">
      <img class="movie-poster"
           src="{safe_poster}"
           onerror="this.onerror=null;this.src='{safe_placeholder}';"
           alt="Poster" />
      <div class="movie-title">{safe_title}</div>
      <div class="meta-row">
        {score_html}
        {year_html}
        {imdb_html}
      </div>
      <hr class="divider">
      <div class="meta-row">{genres_html}</div>
      <hr class="divider">
      <div class="plot-text">{safe_description}</div>
      {director_html}
    </div>
    </body>
    </html>
    """
    components.html(card_html, height=520, scrolling=False)


# ── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ──
#  GRID RENDER HELPER
# ── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ──

def display_recommendations_grid(recs_df: pd.DataFrame, custom_badge: str = None):
    if recs_df.empty:
        st.write("No recommendations available for this section.")
        return
        
    cols_per_row = 4
    total = len(recs_df)
    rows = (total + cols_per_row - 1) // cols_per_row

    idx = 0
    for _ in range(rows):
        cols = st.columns(cols_per_row)
        for col in cols:
            if idx >= total:
                break
            row = recs_df.iloc[idx]
            details = fetch_movie_details(row["title"])
            with col:
                render_movie_card(row, details, custom_badge)
            idx += 1


# ── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ──
#  MAIN APP
# ── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ──

def main():
    movies_df = load_movies()
    model = load_model()
    all_genres = get_all_genres(movies_df)

    st.markdown('<h1 class="hero-title">🎬 CineMatch</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Hybrid Movie Recommendation Engine — '
        'Collaborative Filtering · Content-Based · Neighborhood Similarities</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## ⚙️ Controls")
        st.markdown("---")

        user_type = st.radio(
            "👤 Who are you?",
            options=["🎟️ Existing User", "🆕 New User (Sign Up)"],
            index=0,
        )
        is_new_user = user_type.startswith("🆕")

        st.markdown("---")

        user_id_input = None
        selected_genres = []
        match_mode = "any"

        if not is_new_user:
            user_id_input = st.number_input(
                "🔢 Enter Your User ID",
                min_value=1, max_value=610, value=1, step=1,
                help="Valid IDs for this model: 1 – 609",
            )
        else:
            st.markdown("##### 🎭 Select Genres You Love")
            selected_genres = st.multiselect(
                "Pick one or more genres",
                options=all_genres,
                default=["Action", "Comedy"],
            )
            match_mode = st.radio(
                "Filter logic",
                options=["any", "all"],
                format_func=lambda x: "Match ANY genre" if x == "any" else "Match ALL genres",
            )

        st.markdown("---")
        top_n = st.slider(
            "🎯 Top N Recommendations",
            min_value=4, max_value=20, value=8, step=4,
        )

        st.markdown("---")
        discover_btn = st.button("🚀 Discover Movies")

    if not discover_btn:
        st.markdown("""
        <div class="info-box">
            👈 <b>Use the sidebar</b> to select your preferences, then click
            <b>Discover Movies</b> to get personalized recommendations.
            <br><br>
            🎟️ <b>Existing Users</b> — enter your User ID for full collaborative predictions alongside item and user neighborhood matches.<br>
            🆕 <b>New Users</b> — pick genres you love and we'll resolve the cold-start issue using metadata profiling.
        </div>
        """, unsafe_allow_html=True)
        return

    with st.spinner("🔍 Calculating recommendations & fetching movie media…"):
        if not is_new_user:
            # Generate predictions for all three existing user contexts
            recs_cf = recommend_for_existing_user(model, movies_df, user_id_input, top_n)
            recs_sim_users = recommend_via_similar_users(model, movies_df, user_id_input, top_n)
            recs_sim_items, favorite_title = recommend_similar_items(model, movies_df, user_id_input, top_n)

            if recs_cf.empty:
                st.error(f"❌ User ID **{user_id_input}** was not found. Try IDs between 1 and 609.")
                return

            st.markdown(f'<div class="section-label">🎟️ Recommendations Dashboard for User #{user_id_input}</div>', unsafe_allow_html=True)
            
            # Sub-Tab Layout for existing user contexts
            tab1, tab2, tab3 = st.tabs([
                "🎯 Optimized Predictions", 
                "👥 Similar Users Watched", 
                "🍿 Because You Liked"
            ])
            
            with tab1:
                st.caption("Pure Matrix Factorization: Hidden traits matching your profile metrics.")
                display_recommendations_grid(recs_cf)
                
            with tab2:
                st.caption("Neighborhood Context: Movies loved by people with identical tastes to yours.")
                display_recommendations_grid(recs_sim_users, custom_badge="👥 Taste Match")
                
            with tab3:
                st.caption(f"Item Vector Context: Similar characteristics to your top-rated film: **{favorite_title}**")
                display_recommendations_grid(recs_sim_items, custom_badge="🎬 Item Match")

        else:
            if not selected_genres:
                st.warning("⚠️ Please select at least one genre to continue.")
                return

            recs = recommend_cold_start(movies_df, selected_genres, top_n, match_mode)
            if recs.empty:
                st.warning("No movies found for the selected genre combination. Try 'any' mode.")
                return

            genre_str = ", ".join(selected_genres)
            st.markdown(
                f'<div class="section-label">🎭 Top {len(recs)} movies for genres: {genre_str}</div>',
                unsafe_allow_html=True,
            )
            display_recommendations_grid(recs)

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center;color:#444466;font-size:0.8rem;">'
        'CineMatch · Powered by SVD & Cosine Similarity Embeddings · OMDb API'
        '</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
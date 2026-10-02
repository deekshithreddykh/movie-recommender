import streamlit as st
import pandas as pd
import pickle


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, #18233d 0%, transparent 35%),
        radial-gradient(circle at top right, #32182f 0%, transparent 30%),
        #080b12;
    color: white;
}

/* Main container */

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* Hero */

.hero {
    padding: 50px 10px 35px 10px;
}

.hero h1 {
    font-size: 58px;
    font-weight: 800;
    margin-bottom: 10px;
    letter-spacing: -2px;
}

.hero span {
    background: linear-gradient(
        90deg,
        #ff4b6e,
        #ff8a4c,
        #ffc857
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #a7adbd;
    font-size: 18px;
    max-width: 650px;
}


/* Movie card */

.movie-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 20px;
    transition: 0.3s;
}

.movie-card:hover {
    border-color: rgba(255,255,255,0.2);
    transform: translateY(-3px);
}

.movie-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 8px;
}

.movie-meta {
    color: #a7adbd;
    font-size: 14px;
    line-height: 1.7;
}

.rating {
    color: #ffc857;
    font-weight: 700;
}


/* Section */

.section-title {
    font-size: 26px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 18px;
}


/* Metrics */

.metric-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 18px;
    text-align: center;
}

.metric-number {
    font-size: 26px;
    font-weight: 800;
}

.metric-label {
    color: #9299aa;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    with open(
        "model/recommender.pkl",
        "rb"
    ) as file:

        return pickle.load(file)


model = load_model()

movies = model["movies"]
similarity = model["similarity"]
movie_indices = model["movie_indices"]


# ==========================================
# RECOMMENDATION FUNCTION
# ==========================================

def recommend(movie_name, number=6):

    if movie_name not in movie_indices:

        return pd.DataFrame()

    index = movie_indices[movie_name]

    similarity_scores = list(
        enumerate(similarity[index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    movie_indices_list = [
        item[0]
        for item in similarity_scores[1:number+1]
    ]

    recommendations = movies.iloc[
        movie_indices_list
    ].copy()

    recommendations["Similarity"] = [
        round(
            similarity_scores[i+1][1] * 100,
            1
        )
        for i in range(len(recommendations))
    ]

    return recommendations


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("## 🎬 CineMatch")

    st.caption(
        "AI Movie Recommendation System"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Discover",
            "Movie Explorer",
            "About"
        ]
    )

    st.divider()

    st.caption(
        "Built with Python • Pandas • Scikit-learn • Streamlit"
    )


# ==========================================
# DISCOVER PAGE
# ==========================================

if page == "Discover":

    st.markdown("""
    <div class="hero">

    <h1>Find your next <span>favorite movie.</span></h1>

    <p>
    Discover movies based on genres, directors and
    actors using a content-based recommendation engine.
    </p>

    </div>
    """, unsafe_allow_html=True)


    # --------------------------------------
    # Statistics
    # --------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-box">
        <div class="metric-number">5.6K+</div>
        <div class="metric-label">Movies</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-box">
        <div class="metric-number">376</div>
        <div class="metric-label">Genres</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-box">
        <div class="metric-number">2.4K+</div>
        <div class="metric-label">Directors</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-box">
        <div class="metric-number">AI</div>
        <div class="metric-label">Recommendations</div>
        </div>
        """, unsafe_allow_html=True)


    st.markdown(
        '<div class="section-title">Choose a movie</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------
    # Movie selector
    # --------------------------------------

    movie_list = sorted(
        movies["Name"].dropna().unique()
    )

    selected_movie = st.selectbox(
        "Search for a movie",
        movie_list
    )


    # --------------------------------------
    # Selected movie
    # --------------------------------------

    selected_data = movies[
        movies["Name"] == selected_movie
    ].iloc[0]


    st.markdown(
        '<div class="section-title">Selected Movie</div>',
        unsafe_allow_html=True
    )


    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            f"""
            <div class="movie-card">

            <div class="movie-title">
            {selected_data['Name']}
            </div>

            <div class="movie-meta">

            <span class="rating">
            ⭐ {selected_data['Rating']}
            </span>

            <br>

            🎭 {selected_data['Genre']}

            <br>

            🎬 {selected_data['Director']}

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c2:

        st.markdown(
            f"""
            <div class="movie-card">

            <div class="movie-title">
            Movie Details
            </div>

            <div class="movie-meta">

            📅 Year: {selected_data['Year']}

            <br>

            ⏱ Duration: {selected_data['Duration']} min

            <br>

            👁 Votes: {selected_data['Votes']}

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c3:

        st.markdown(
            f"""
            <div class="movie-card">

            <div class="movie-title">
            Cast
            </div>

            <div class="movie-meta">

            👤 {selected_data['Actor 1']}

            <br>

            👤 {selected_data['Actor 2']}

            <br>

            👤 {selected_data['Actor 3']}

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------
    # Recommendations
    # --------------------------------------

    st.markdown(
        '<div class="section-title">✨ Recommended for you</div>',
        unsafe_allow_html=True
    )


    recommendations = recommend(
        selected_movie,
        6
    )


    cols = st.columns(3)


    for index, (_, movie) in enumerate(
        recommendations.iterrows()
    ):

        with cols[index % 3]:

            st.markdown(
                f"""
                <div class="movie-card">

                <div class="movie-title">
                {movie['Name']}
                </div>

                <div class="movie-meta">

                ⭐ <span class="rating">
                {movie['Rating']}
                </span>

                <br>

                🎭 {movie['Genre']}

                <br>

                📅 {movie['Year']}

                <br>

                🎬 {movie['Director']}

                <br>

                🔥 Similarity:
                {movie['Similarity']}%

                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ==========================================
# MOVIE EXPLORER
# ==========================================

elif page == "Movie Explorer":

    st.markdown(
        '<div class="section-title">🎞 Movie Explorer</div>',
        unsafe_allow_html=True
    )

    search = st.text_input(
        "Search movie"
    )


    filtered = movies.copy()

    if search:

        filtered = filtered[
            filtered["Name"]
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]


    st.write(
        f"Showing {len(filtered)} movies"
    )


    st.dataframe(
        filtered[
            [
                "Name",
                "Year",
                "Genre",
                "Rating",
                "Votes",
                "Director"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# ABOUT
# ==========================================

elif page == "About":

    st.markdown(
        '<div class="section-title">About CineMatch</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ### 🎬 CineMatch

    CineMatch is a content-based movie recommendation
    system built using Python and Machine Learning techniques.

    ### How it works

    1. Movie information is collected.
    2. The dataset is cleaned.
    3. Genre, director and cast information are combined.
    4. TF-IDF converts the text into numerical vectors.
    5. Cosine similarity calculates movie similarity.
    6. Similar movies are recommended to the user.

    ### Technologies

    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - TF-IDF
    - Cosine Similarity
    - Streamlit
    """)

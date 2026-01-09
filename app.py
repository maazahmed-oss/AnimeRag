import streamlit as st
import pandas as pd
from groq import Groq
import os

# --------------------------------------------------
# 1. PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AnimeRAG",
    page_icon="🎌",
    layout="centered"
)

# --------------------------------------------------
# 2. UI STYLING
# --------------------------------------------------
st.markdown("""
<style>
.main-title {
    font-size: 3.2rem;
    font-weight: 800;
    text-align: center;
    margin-top: -40px;
}
.sub-text {
    text-align: center;
    color: #555;
    font-size: 1.2rem;
    margin-bottom: 30px;
    font-style: italic;
}
[data-testid="stSidebar"] {
    background-color: #F0F2F6;
}
.stChatMessage {
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 3. GROQ API
# --------------------------------------------------
if "GROQ_API_KEY" not in st.secrets:
    st.error("⚠️ GROQ_API_KEY missing in Streamlit secrets.")
    st.stop()

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# --------------------------------------------------
# 4. FIND CSV FILE (CASE SAFE)
# --------------------------------------------------
def find_csv_file():
    for f in os.listdir("."):
        if f.lower() == "anime.csv":
            return f
    return None

CSV_FILE = find_csv_file()

if not CSV_FILE:
    st.error("❌ Anime.csv not found in repository.")
    st.stop()

# --------------------------------------------------
# 5. LOAD CSV (EXACT COLUMN MATCH)
# --------------------------------------------------
@st.cache_data
def load_anime_data(csv_file):
    df = pd.read_csv(csv_file)

    # Explicit mapping (your real columns)
    df = df[['Name', 'Rating']].dropna()

    # Standardize column names
    df.columns = ['name', 'rating']

    # Convert to text for LLM
    anime_text = ""
    for _, row in df.iterrows():
        anime_text += f"Anime: {row['name']} | Rating: {row['rating']}\n"

    return df, anime_text


anime_df, anime_text = load_anime_data(CSV_FILE)

# --------------------------------------------------
# 6. SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🎌 AnimeRAG")
    st.caption("v1.2 • Real Anime Dataset")
    st.divider()

    st.markdown("### 📊 Dataset Info")
    st.info(f"""
- File: **{CSV_FILE}**
- Total Anime: **{len(anime_df)}**
- Columns Used: Name, Rating
""")

    st.divider()
    if st.button("🔄 New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------
# 7. MAIN HEADER
# --------------------------------------------------
st.markdown('<p class="main-title">AnimeRAG</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Ask anything about anime ratings</p>', unsafe_allow_html=True)

# --------------------------------------------------
# 8. CHAT INIT
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "👋 **Assalam o Alaikum!**\n\n"
            "I am **AnimeRAG**.\n"
            "You can ask about anime ratings, top-rated anime, or filters like rating > 9."
        )
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------
# 9. USER INPUT + STREAMING
# --------------------------------------------------
if prompt := st.chat_input("Ask: rating of Naruto, top anime, rating > 9..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        box = st.empty()
        answer = ""

        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": f"""
You are AnimeRAG.
Answer ONLY from this anime dataset.
If anime not found, say clearly.

Dataset:
{anime_text[:15000]}
"""
                },
                {"role": "user", "content": prompt}
            ],
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                answer += chunk.choices[0].delta.content
                box.markdown(answer + "▌")

        box.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

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
# 2. PROFESSIONAL UI STYLING
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
# 3. GROQ CONNECTION
# --------------------------------------------------
if "GROQ_API_KEY" not in st.secrets:
    st.error("⚠️ GROQ_API_KEY missing in Streamlit secrets.")
    st.stop()

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# --------------------------------------------------
# 4. LOAD ANIME CSV (RAG DATA)
# --------------------------------------------------
CSV_FILE = "Anime.csv"

@st.cache_data
def load_anime_data():
    if not os.path.exists(CSV_FILE):
        return None, None

    df = pd.read_csv(CSV_FILE)

    # Basic cleanup
    df = df[['name', 'rating']].dropna()

    # Convert to text for LLM
    anime_text = ""
    for _, row in df.iterrows():
        anime_text += f"Anime: {row['name']} | Rating: {row['rating']}\n"

    return df, anime_text

anime_df, anime_text = load_anime_data()

if anime_df is None:
    st.error("❌ anime.csv file not found. Please upload it.")
    st.stop()

# --------------------------------------------------
# 5. SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🎌 AnimeRAG")
    st.caption("v1.0 • Anime Intelligence System")
    st.divider()

    st.markdown("### 📊 Dataset Info")
    st.info(f"""
- Total Anime: **{len(anime_df)}**
- Data Source: CSV
- Fields: Name, Rating
""")

    st.divider()
    if st.button("🔄 New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------
# 6. MAIN HEADER
# --------------------------------------------------
st.markdown('<p class="main-title">AnimeRAG</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Ask anything about anime ratings</p>', unsafe_allow_html=True)

# --------------------------------------------------
# 7. CHAT SESSION INIT
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 **Assalam o Alaikum!**\n\nI am **AnimeRAG**. You can ask me about anime names, ratings, or top-rated anime from the dataset."
        }
    ]

# --------------------------------------------------
# 8. DISPLAY CHAT HISTORY
# --------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------
# 9. USER INPUT + GROQ STREAMING
# --------------------------------------------------
if prompt := st.chat_input("Ask: rating of Naruto, top anime, rating > 9..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_box = st.empty()
        full_response = ""

        try:
            stream = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": f"""
You are AnimeRAG.
Answer ONLY from the given anime dataset.
If anime is not found, say: "Anime not found in dataset."

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
                    full_response += chunk.choices[0].delta.content
                    response_box.markdown(full_response + "▌")

            response_box.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

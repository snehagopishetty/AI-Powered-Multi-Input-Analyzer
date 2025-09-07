import streamlit as st
from scripts import dataIngestion
from scripts import collecting
from scripts.extra_features import extract_entities, analyze_sentiment, export_summary_to_pdf
import os
import tempfile
from collections import defaultdict
import pandas as pd


# Set page configuration
st.set_page_config(page_title="NewsSense", layout="wide")

# Custom CSS
st.markdown("""
<style>
    body {
        background-color: #f4f6f9;
        font-family: 'Roboto', sans-serif;
        color: #333333;
    }

    .stButton > button {
        background-color: #007BFF !important;
        color: #ffffff !important;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        border-radius: 8px;
        transition: background-color 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #0056b3 !important;
    }

    .output-box {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-left: 5px solid #17a2b8;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }

    .output-box h3 {
        color: #17a2b8;
        margin-bottom: 10px;
        font-size: 20px;
        font-weight: 600;
    }

    .output-box p {
        color: #333333;
        font-size: 16px;
        line-height: 1.6;
    }

    .download-btn {
        background-color: #ffc107;
        color: #212529;
        padding: 12px 24px;
        border: none;
        border-radius: 8px;
        font-size: 16px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }

    .download-btn:hover {
        background-color: #e0a800;
    }

    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: #007BFF;
        margin-bottom: 30px;
    }

    .stRadio > div {
        padding: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# === Named Entity Renderer ===
def render_entities(entities):
    if not entities:
        st.warning("No named entities found.")
        return

    grouped_entities = defaultdict(list)
    for text, label in entities:
        grouped_entities[label].append(text)

    # Define colors for each label
    colors = {
        "PERSON": "#FF6F61",
        "ORG": "#6B5B95",
        "GPE": "#88B04B",
        "LOC": "#F7CAC9",
        "DATE": "#92A8D1",
        "MONEY": "#F4A261",
        "TIME": "#2A9D8F",
        # Add more as needed
    }

    with st.expander("🏷️ Named Entities"):
        for label, texts in sorted(grouped_entities.items()):
            color = colors.get(label, "#999999")  # Default grey
            st.markdown(
                f"<div style='background-color:{color}; padding:10px; border-radius:6px; color:white; margin-top:15px;'>"
                f"<strong>{label}</strong> &nbsp;&nbsp;<span style='font-size:14px;'>(Count: {len(set(texts))})</span>"
                "</div>",
                unsafe_allow_html=True
            )
            df = pd.DataFrame(sorted(set(texts)), columns=["Entity"])
            st.table(df)


# === Title ===
st.markdown('<h1 class="main-title">📰 NewsSense: Smart News Analyzer</h1>', unsafe_allow_html=True)

language = st.sidebar.selectbox("Input Language", ["English", "Hindi", "Telugu"])

# === Sidebar Input Options ===
st.sidebar.header("📥 Input Options")
input_option = st.sidebar.radio("Choose Input Type:", ("URL", "PDF", "Text", "YouTube" , "image"))
article_text = ""

# === Input Handlers ===
if input_option == "URL":
    url = st.sidebar.text_input("Enter Article URL")
    if st.sidebar.button("Extract and Analyze"):
        if url:
            article_text = dataIngestion.extract_text_from_url(url)
elif input_option == "PDF":
    pdf_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    if pdf_file is not None and st.sidebar.button("Extract and Analyze"):
        article_text = dataIngestion.extract_text_from_pdf(pdf_file)
elif input_option == "YouTube":
    yt_url = st.sidebar.text_input("Enter YouTube Video URL")
    if st.sidebar.button("Extract Transcript and Analyze"):
        if yt_url:
            article_text = dataIngestion.extract_text_from_youtube(yt_url)
elif input_option == "Upload Image":
    image_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if st.sidebar.button("Process Image"):
        if image_file is not None:
            article_text = dataIngestion.extract_text_from_image(image_file)
        else:
            st.sidebar.warning("Please upload an image.")
else:
    article_text = st.sidebar.text_area("Paste your news article text here")
    if st.sidebar.button("Analyze"):
        if not article_text.strip():
            st.sidebar.warning("Please enter some text")

show_notes = st.sidebar.checkbox("📝 Take Notes", value=False)

if show_notes:
    user_input_note = st.sidebar.text_area(
        "Your notes:",
        value=st.session_state.get("user_notes", ""),
        height=150,
        key="note_sidebar_input"
    )

    if st.sidebar.button("💾 Save Notes"):
        st.session_state.user_notes = user_input_note
        st.sidebar.success("Notes saved for this session!")

    if st.session_state.get("user_notes", "").strip():
        st.sidebar.download_button(
            "📥 Download Notes",
            data=st.session_state.user_notes,
            file_name="article_notes.txt",
            mime="text/plain"
        )


# === Main NLP Processing ===
if article_text:
    if language != 'English':
        with st.expander("🌐 Translated Input"):
                article_text = dataIngestion.translate_to_english(article_text, src_lang='hi' if language == 'Hindi' else 'te')
   
    with st.spinner("🔍 Analyzing the article... Please wait."):

        category = collecting.predict_category(article_text)
        summary_abs = collecting.get_summary(article_text)
        lda_topics = collecting.topic_modeling(article_text)
        entities = extract_entities(article_text)
        sentiment = analyze_sentiment(article_text)

        st.session_state.update({
            "category": category,
            "summary": summary_abs,
            "lda_topics": lda_topics,
            "sentiment": sentiment,
            "entities": entities,
        })

    st.success("✅ Analysis complete! Scroll down to view the results.")

    # === Extracted Text Preview ===
    with st.expander("📄 Extracted Text (First 1000 Characters)"):
        st.write(article_text[:1000])

    # === Display Results ===
    st.markdown(f'''
        <div class="output-box">
            <h3>📌 Predicted Category</h3>
            <p>{category}</p>
        </div>
    ''', unsafe_allow_html=True)

    st.markdown(f'''
        <div class="output-box">
            <h3>📝 Summary</h3>
            <p>{summary_abs}</p>
        </div>
    ''', unsafe_allow_html=True)

    if lda_topics:
        keywords_html = "".join(
            f"<span style='background:#17a2b8; color:#fff; padding:4px 8px; margin:2px; border-radius:4px;'>{word}</span>"
            for word, _ in lda_topics[0][1])
        st.markdown(f"<div class='output-box'><h3>🔑 Keywords</h3><p>{keywords_html}</p></div>", unsafe_allow_html=True)
    else:
        st.warning("No keywords found.")


    # === Display Named Entities as Tables Grouped by Label ===
    render_entities(entities)

    label, confidence = dataIngestion.detect_fake_news(article_text)
    st.subheader("Fake News Detection")
    if label == "FAKE":
        st.error(f"Prediction: {label} (Confidence: {confidence})")
    else:
        st.success(f"Prediction: {label} (Confidence: {confidence})")

    # === Display Sentiment ===
    st.markdown(f"""
        <div style="background-color:#e9f7ef; border-radius:10px; text-align:center; padding:30px; margin-top:30px;">
            <h3 style="color:#2e7d32;">Overall Sentiment</h3>
            <p style="font-size:22px; font-weight:bold; color:#2e7d32;">{sentiment}</p>
        </div>
    """, unsafe_allow_html=True)

# === PDF Generation ===
custom_filename = st.sidebar.text_input("✏️ Enter custom filename for the PDF (without .pdf)", value="news_summary")

if st.sidebar.button("📄 Generate PDF"):
    if not ("category" in st.session_state and "summary" in st.session_state):
        st.sidebar.error("Please analyze an article first before generating PDF.")
    else:
        with st.spinner("Generating PDF..."):
            lda_topics = st.session_state.get("lda_topics", [])
            keywords_list = []
            if lda_topics:
                keywords_list = [word for word, _ in lda_topics[0][1]]

            custom_name = custom_filename.strip() or "news_summary"

            with tempfile.TemporaryDirectory() as tmpdirname:
                pdf_filename = f"{custom_name}.pdf"
                pdf_path = os.path.join(tmpdirname, pdf_filename)

                result_path = export_summary_to_pdf(
                    pdf_path,
                    st.session_state["category"],
                    st.session_state["summary"],
                    keywords_list,
                    st.session_state["sentiment"],
                    st.session_state["entities"]
                )

                if result_path and os.path.exists(result_path):
                    with open(result_path, "rb") as f:
                        st.sidebar.download_button(
                            label="📥 Download PDF Summary",
                            data=f,
                            file_name=pdf_filename,
                            mime="application/pdf",
                            use_container_width=True,
                            key="download_pdf"
                        )
                else:
                    st.sidebar.error("PDF generation failed or file not found.")

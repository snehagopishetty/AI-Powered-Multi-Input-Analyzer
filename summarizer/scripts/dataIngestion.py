from newspaper import Article
from newspaper.article import ArticleException
import fitz
from youtube_transcript_api import YouTubeTranscriptApi
from PIL import Image
import pytesseract
import re
from deep_translator import GoogleTranslator
from transformers import pipeline

def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except ArticleException as e:
        print(f"Failed to download article: {e}")
        return ""

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_youtube(url):
    try: 
        if "v=" in url:
            video_id = url.split("v=")[-1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[-1].split("?")[0]
        else:
            return "Invalid YouTube URL format."
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        full_text = " ".join([entry["text"] for entry in transcript])
        return full_text

    except Exception as e:
        return f"Failed to retrieve transcript: {str(e)}"
    
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def extract_text_from_image(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text

def translate_to_english(text: str, src_lang: str) -> str:
    try:
        # Split the text into paragraphs or sentences
        chunks = re.split(r'(?<=[।.!?])\s+', text.strip())  
        translated_chunks = []
        for chunk in chunks:
            if chunk.strip():
                translated = GoogleTranslator(source=src_lang, target='en').translate(chunk)
                translated_chunks.append(translated)
        return " ".join(translated_chunks)
    except Exception as e:
        return f"Translation Error: {str(e)}"
    
fake_news_model = pipeline("text-classification", model="Pulk17/Fake-News-Detection")     

label_map = {
    "LABEL_0": "REAL",
    "LABEL_1": "FAKE"
}

def detect_fake_news(text):
    result = fake_news_model(text[:512])[0]
    label = label_map.get(result['label'], result['label'])
    score = result['score']
    return label, round(score, 3)


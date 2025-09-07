import spacy
from textblob import TextBlob
import os
from fpdf import FPDF

nlp = spacy.load("en_core_web_sm")
def extract_entities(text):
    doc = nlp(text)
    unique_entities = set((ent.text.strip(), ent.label_) for ent in doc.ents)
    return list(unique_entities)



def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

def sanitize_text(text):
    return text.encode("latin-1", "replace").decode("latin-1")

# Export to PDF (fpdf, Latin-1 safe)
def export_summary_to_pdf(filename, category, summary, keywords, sentiment, entities):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, sanitize_text("NewsSense Summary Report"), ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.ln(5)
    pdf.cell(0, 10, sanitize_text(f"Category: {category}"), ln=True)
    pdf.ln(5)

    pdf.multi_cell(0, 10, sanitize_text(f"Summary:\n{summary}"))
    pdf.ln(5)

    pdf.cell(0, 10, sanitize_text(f"Sentiment: {sentiment}"), ln=True)
    pdf.ln(5)

    pdf.multi_cell(0, 10, sanitize_text(f"Keywords: {', '.join(keywords)}"))
    pdf.ln(5)

    pdf.cell(0, 10, sanitize_text("Named Entities:"), ln=True)
    for ent_text, ent_type in entities:
        # Replace bullet (•) with hyphen (-)
        pdf.cell(0, 10, sanitize_text(f"- {ent_text} ({ent_type})"), ln=True)

    output_path = os.path.join("exports", filename)
    os.makedirs("exports", exist_ok=True)
    pdf.output(output_path)
    return output_path
from transformers import pipeline
from transformers import pipeline, BartTokenizer

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def abstractive_summary(text, max_length=130, min_length=30):
    max_input_tokens = 1024  # max tokens BART supports
    
    # Properly encode with truncation
    inputs = tokenizer(text, max_length=max_input_tokens, truncation=True, return_tensors="pt")
    
    # Decode truncated input tokens back to string
    truncated_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    
    summary_list = summarizer(truncated_text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary_list[0]['summary_text']

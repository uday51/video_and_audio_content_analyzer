import openai
import os
from moviepy.editor import VideoFileClip
import requests
from transformers import pipeline, AutoTokenizer
import torch
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

# Set up NLTK
nltk_data_path = '/usr/local/nltk_data'
os.makedirs(nltk_data_path, exist_ok=True)
nltk.download('punkt', download_dir=nltk_data_path)
nltk.data.path.append(nltk_data_path)

# Initialize models with tokenizers
summarizer_model = "falconsai/text_summarization"
summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_model)
summarizer = pipeline('summarization', model=summarizer_model, tokenizer=summarizer_tokenizer)

sentiment_model = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model)
sentiment = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)

def chunk_text(full_text, max_tokens=400):  # Reduced to allow room for special tokens
    sentences = sent_tokenize(full_text)
    chunks = []
    current_chunk = ""
    current_len = 0
    
    for sentence in sentences:
        # Tokenize sentence to get accurate length
        tokens = sentiment_tokenizer.tokenize(sentence)
        if current_len + len(tokens) <= max_tokens:
            current_chunk += " " + sentence
            current_len += len(tokens)
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_len = len(tokens)
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def process_transcript(full_text):
    chunks = chunk_text(full_text)
    print(f"🔹 Total chunks: {len(chunks)}")

    all_summaries = []
    all_sentiments = []

    for i, chunk in enumerate(chunks):
        try:
            # Summarization
            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            print(f"🧠 Summary {i+1}: {summary}")
            all_summaries.append(summary)

            # Sentiment Analysis
            sentiment_result = sentiment(chunk, truncation=True, max_length=512)[0]
            print(f"❤️ Sentiment {i+1}: {sentiment_result['label']} ({sentiment_result['score']:.2f})")
            all_sentiments.append(sentiment_result)
        except Exception as e:
            print(f"Error processing chunk {i+1}: {str(e)}")
            continue
    
    return all_summaries, all_sentiments



client=openai.OpenAI(api_key='')

HF_TOKEN = ""
headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}
video_path='video.mp4'
extracted_audio="audio.mp3"

video=VideoFileClip(video_path)
video.audio.write_audiofile(extracted_audio)


with open(extracted_audio,'rb') as filepath:
  trancript=client.audio.transcriptions.create(
    model='whisper-1',
    file=filepath
    )
    
print(trancript.text)
output_text=trancript.text

# Process the transcript
summaries, sentiments = process_transcript(output_text)
print("Completed")

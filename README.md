# # 🎥 Video Transcript Summarizer and Sentiment Analyzer

This project extracts audio from a video file, transcribes the spoken content using OpenAI Whisper, summarizes the transcript using a Hugging Face transformer model, and performs sentiment analysis.

---

## 🚀 Features

- Extract audio from video using `moviepy`
- Transcribe speech to text using `OpenAI Whisper`
- Summarize large transcripts using `falconsai/text_summarization`
- Perform sentiment analysis using `distilbert-base-uncased-finetuned-sst-2-english`
- Breaks long transcripts into manageable chunks
- Displays summaries and sentiment scores for each chunk

---

## 📦 Requirements

Install the dependencies using pip:

```bash
pip install openai moviepy torch transformers nltk
🔐 API Keys
Make sure you have:

OpenAI API Key

Hugging Face Token

You can set them in your script like this:

python
Copy
Edit
OPENAI_API_KEY = "your_openai_api_key"
HF_TOKEN = "your_huggingface_token"
📁 Input
Place your video file in the project directory. For example:

python
Copy
Edit
video_path = "video.mp4"
🧠 How It Works
Extracts audio from the video file.

Transcribes the audio to text using OpenAI's Whisper model.

Tokenizes the transcript into sentence-based chunks.

Summarizes each chunk using a transformer model.

Performs sentiment analysis for each chunk.

Displays all summaries and sentiments.

🧪 Example Output
bash
Copy
Edit
🔹 Total chunks: 4
🧠 Summary: The speaker discusses the team's performance and challenges in recent matches...
❤️ Sentiment: POSITIVE (0.91)
...
🛠 Tech Stack
Python

OpenAI Whisper

Hugging Face Transformers

MoviePy

NLTK

📌 Notes
Make sure punkt tokenizer is downloaded for NLTK:

python
Copy
Edit
import nltk
nltk.download('punkt')
Set nltk.data.path only if you’re using a custom location.

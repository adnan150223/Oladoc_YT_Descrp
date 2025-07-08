import os
import io
import tempfile
import subprocess
import streamlit as st
import google.generativeai as genai
from google.cloud import speech
from pydub import AudioSegment
import re

# --- Remove the dotenv code because we don't need it anymore ---
# from dotenv import load_dotenv
# load_dotenv()

# --- Use Streamlit Secrets to access credentials --- 
google_credentials_json = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]

# Write the JSON credentials to a temporary file that Google API can use
with open("/tmp/credentials.json", "w") as json_file:
    json_file.write(google_credentials_json)

# Set the credentials environment variable for Google APIs
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/credentials.json"

# --- Configuration ---
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

# --- Helper Functions ---
# Compress video to smaller size
def compress_video(input_path, output_path):
    command = ["ffmpeg", "-i", input_path, "-vcodec", "libx264", "-crf", "28", output_path]
    subprocess.run(command, check=True)

# Split video into smaller chunks based on size
def split_video(input_path, chunk_size_mb=200):
    output_files = []
    # Get the duration of the video
    command = ["ffmpeg", "-i", input_path, "-f", "null", "-"]
    result = subprocess.run(command, capture_output=True, text=True)
    duration = float(result.stderr.split("Duration: ")[1].split(",")[0].split(":")[2])

    # Split the video into chunks of approximately chunk_size_mb
    chunk_count = int(duration // (chunk_size_mb * 8 / 1000))  # approx. size in seconds
    for i in range(chunk_count):
        output_file = f"chunk_{i + 1}.mp4"
        command = ["ffmpeg", "-i", input_path, "-ss", str(i * chunk_size_mb), "-t", str(chunk_size_mb), output_file]
        subprocess.run(command, check=True)
        output_files.append(output_file)

    return output_files

# Extract audio from video
def extract_audio(video_path, output_path):
    command = ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", output_path]
    subprocess.run(command, check=True)

# Split audio into smaller chunks
def split_audio(audio_path, chunk_length_ms=30000):
    audio = AudioSegment.from_wav(audio_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i+chunk_length_ms]
        chunk_path = f"chunk_{i//chunk_length_ms}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks

# Transcribe audio using Google Speech-to-Text
def transcribe_google(audio_path):
    client = speech.SpeechClient()
    with io.open(audio_path, "rb") as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="ur-PK",
        alternative_language_codes=["en-US"]
    )
    response = client.recognize(config=config, audio=audio)
    return " ".join([r.alternatives[0].transcript for r in response.results])

# Translate text using Gemini AI
def translate_with_gemini(urdu_text):
    prompt = f"""
You are an expert translator with a deep understanding of both Urdu and English. Your task is to translate the following Urdu text into English, ensuring that the meaning remains clear and accurate. The text you're translating may have issues, such as:
1. *Grammatical errors* in the original Urdu.
2. *Transcription mistakes* that result in unclear or missing context.
3. *Nonsensical or irrelevant phrases* due to poor transcription.

Your goal is to:
- *Translate the text accurately*, maintaining its core meaning and technical accuracy (particularly in medical or specialized contexts).
- *Fix any grammatical errors* in the Urdu text during translation.
- If the text is unclear or incomplete, *make educated guesses* to provide a reasonable translation, and *highlight any ambiguities* in square brackets, e.g., "[unclear part]".
- *Ignore any irrelevant or nonsensical phrases* (like promotional content or transcription artifacts), and *rephrase them appropriately* if necessary.
- Ensure that *scientific and medical terms* are translated correctly with appropriate terminology.

Here's the Urdu text to translate:

{urdu_text}

If there are sections that are incomplete, unclear, or contain transcription errors, please:
- *Provide the best possible translation* and note the issue with the original text.
- If the translation of a specific term or phrase is unclear, explain your interpretation.

Provide the translated text, fixing any errors and ensuring clarity.
"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

# Generate YouTube description based on text and doctor details
def generate_youtube_description(english_text, doctor_name, specialization, clinic, city, booking_link):
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
        You are a professional content creator for a healthcare platform. Your task is to generate a *YouTube video description* based on the provided text. The description should be clear, engaging, and SEO-optimized for a general audience, with both English and Roman Urdu text included. The description should be simple enough for a broad audience to understand.

        The text you need to work with is the following:

        *English Text:*
        {english_text}

        Your output should be a well-structured YouTube description with the following sections without mention any heading write in the form of the paragraph just symptom should be write in the bullets:

        Create a bilingual title (English + Roman Urdu) in 80 to 90 characters that is attention-grabbing and summarizes the video topic. Use simple language that is relatable to the local audience. Use the title words in the rest of the description.
        The first 2-3 sentences should briefly summarize the key points of the video. Mention why it’s important and grab the viewer's attention. Use simple, engaging English to make it relatable.
        Mention the doctor's name{doctor_name}, specialization{specialization}, and the clinic/hospital{clinic} they practice at city name{city}. Provide the booking link{booking_link} for consultations if applicable. Keep it concise and clear.
        Use bullet points or a numbered list to mention the key topics covered in the video. Focus on making it simple and engaging, using simple English wherever applicable.
        Share actionable advice or solutions that can help viewers.
        Include a strong call to action encouraging viewers to book a consultation or contact the doctor. Provide a booking link or phone number. Ensure that the CTA uses simple English language to encourage immediate action.
        """
    response = model.generate_content(prompt)
    return response.text

# --- Clean Description Function ---
def clean_description(description):
    lines = description.split('\n')
    cleaned_lines = []
    for line in lines:
        if re.match(r'^\s*[A-Za-z ]+\s*:\s*$', line.strip()):
            continue
        if line.strip().lower() in ['title', 'description', 'youtube description', 'summary', 'doctor details']:
            continue
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines).strip()

# --- Ensure Description Length is 2000 Characters for Large Videos ---
def limit_description_length(description, max_length=2000):
    if len(description) > max_length:
        return description[:max_length]
    return description

# --- Streamlit UI ---
st.set_page_config(page_title="Auto YouTube Description Generator", layout="centered")
st.title("🎬 Auto YouTube Description Generator")
st.markdown("Upload a video file in Urdu and get a YouTube-ready description.")

# --- Doctor's Details Input ---
doctor_name = st.text_input("👨‍⚕ Doctor's Name")
specialization = st.text_input("💼 Specialization")
clinic = st.text_input("🏥 Clinic")
city = st.text_input("📍 City")
booking_link = st.text_input("🔗 Booking Link")

# --- Upload Video After Doctor Details ---
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        tmp_video.write(uploaded_file.read())
        video_path = tmp_video.name

    # Check file size and handle accordingly
    if os.path.getsize(video_path) > 200 * 1024 * 1024:
        st.warning("The video is too large. It will be compressed before further processing.")
        compressed_video_path = "compressed_video.mp4"
        compress_video(video_path, compressed_video_path)
        video_path = compressed_video_path  # Use the compressed video for further processing
        st.success("✅ Video compressed successfully.")

    # If the video is large, split it into smaller chunks
    if os.path.getsize(video_path) > 200 * 1024 * 1024:
        st.warning("The video is large. Splitting it into smaller chunks...")
        chunks = split_video(video_path)
        st.success(f"✅ Video split into {len(chunks)} chunks.")
    else:
        st.success("✅ Video uploaded. Extracting audio...")

    # Extract audio and process
    audio_path = "extracted_stream_audio.wav"
    extract_audio(video_path, audio_path)
    chunks = split_audio(audio_path)

    full_urdu = ""
    for chunk in chunks:
        st.write(f"🔊 Transcribing {chunk}...")
        full_urdu += transcribe_google(chunk) + " "

    st.text_area("📝 Urdu Transcription", full_urdu, height=150)

    st.success("✅ Translating to English...")
    english_text = translate_with_gemini(full_urdu)
    st.text_area("🌍 English Translation", english_text, height=150)

    st.success("🚀 Generating description with Gemini...")
    description = generate_youtube_description(
        english_text, doctor_name, specialization, clinic, city, booking_link
    )
    
    # Limit the description length to 2000 characters for large videos
    if os.path.getsize(video_path) > 200 * 1024 * 1024:
        description = limit_description_length(description, 2000)
    
    # Clean the description
    description = clean_description(description)  # Clean the description
    
    # Ensure the description is in 4 paragraphs
    paragraphs = description.split("\n")
    paragraph_count = 4
    if len(paragraphs) > paragraph_count:
        paragraph_size = len(paragraphs) // paragraph_count
        description = "\n".join(paragraphs[:paragraph_size]) + "\n\n" + "\n".join(paragraphs[paragraph_size:paragraph_size*2]) + "\n\n" + "\n".join(paragraphs[paragraph_size*2:paragraph_size*3]) + "\n\n" + "\n".join(paragraphs[paragraph_size*3:])
    
    st.markdown("### 📄 Final YouTube Description")
    st.text_area("Generated Description", description, height=300)

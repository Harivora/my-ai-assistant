import streamlit as st
st.set_page_config(layout="wide", page_title="My AI Assistant")

import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image
import PyPDF2
import io
import json
from pathlib import Path

# For Audio (basic example)
try:
    import speech_recognition as sr
    from pydub import AudioSegment # For converting audio if needed
except ImportError:
    st.warning("SpeechRecognition or pydub not installed. Audio transcription feature will be limited. `pip install SpeechRecognition pydub`")
    sr = None
    AudioSegment = None

# --- Configuration & Initialization ---
load_dotenv() # Load environment variables from .env file

# Prompt user for Gemini API key if not set in session state
if "GEMINI_API_KEY" not in st.session_state:
    st.session_state.GEMINI_API_KEY = ""

if not st.session_state.GEMINI_API_KEY:
    st.title(" My AI Assistant ✨")
    st.warning("Please enter your Gemini API key to use the assistant.")
    api_key_input = st.text_input("Enter your Gemini API key:", type="password")
    st.markdown("""
    <div style='margin-top: 1em; text-align: center;'>
        <a href='https://aistudio.google.com/app/apikey' target='_blank' style='color: #4f8cff; font-weight: 500; text-decoration: underline;'>Get your Gemini API key here</a>
    </div>
    """, unsafe_allow_html=True)
    if api_key_input:
        # Try to verify the key before saving
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key_input)
            # Try a minimal test call to verify the key
            test_model = genai.GenerativeModel("gemini-1.5-flash-latest")
            test_response = test_model.generate_content("Say hello.")
            if hasattr(test_response, 'text') and test_response.text:
                st.session_state.GEMINI_API_KEY = api_key_input
                st.success("API key verified!")
                st.rerun()
            else:
                st.error("API key could not be verified. Please check your key and try again.")
        except Exception as e:
            st.error(f"API key verification failed: {e}")
    st.stop()

try:
    genai.configure(api_key=st.session_state.GEMINI_API_KEY)
except Exception as e:
    st.error(f"🔴 Error configuring Google API: {e}")
    st.session_state.GEMINI_API_KEY = ""
    st.stop()

# --- Model Initialization ---
# Use a multimodal model that can handle text, images, and has good general capabilities.
try:
    llm_model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest",
        # safety_settings=[...], # Add if specific safety settings are needed
        # generation_config={"temperature": 0.7} # Add if specific generation config is needed
    )
except Exception as e:
    st.error(f"🔴 Error initializing Generative Model: {e}")
    st.session_state.GEMINI_API_KEY = ""
    st.stop()


# --- Session State Management ---
MEMORY_FILE = "chat_memory.json"

def save_chat_history():
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.chat_history, f, ensure_ascii=False, indent=2)

def load_chat_history():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Could not load previous memory: {e}")
    return []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()
if "chat_session" not in st.session_state:
    st.session_state.chat_session = llm_model.start_chat(history=st.session_state.chat_history)

# --- Helper Functions ---

def extract_text_from_pdf(pdf_file_uploader_object):
    """Extracts text from an uploaded PDF file."""
    try:
        # The file uploader object is already a file-like object
        reader = PyPDF2.PdfReader(pdf_file_uploader_object)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"PDF Processing Error: {e}")
        return None

def transcribe_audio(audio_file_uploader_object):
    """Transcribes audio from an uploaded file (basic)."""
    if not sr or not AudioSegment:
        st.warning("Audio transcription libraries (SpeechRecognition, pydub) not fully available.")
        return "Audio transcription libraries not fully available."
    
    recognizer = sr.Recognizer()
    try:
        # audio_file_uploader_object is a BytesIO or similar file-like object from Streamlit
        # Ensure it's in a format pydub can handle or directly by sr.AudioFile
        
        # Attempt to load with pydub to handle various formats and convert to WAV for recognizer
        sound = AudioSegment.from_file(audio_file_uploader_object)
        wav_io = io.BytesIO()
        sound.export(wav_io, format="wav")
        wav_io.seek(0) # Reset pointer to the beginning of the BytesIO object
        
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)
        
        # Using Google Web Speech API (requires internet)
        # TODO: Add language selection support here if desired
        transcribed_text = recognizer.recognize_google(audio_data)
        return transcribed_text
    except sr.UnknownValueError:
        return "Audio could not be understood."
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"
    except Exception as e:
        st.error(f"Audio Transcription Error: {e}")
        return f"Audio Transcription Error: {e}" # Return error message

def extract_text_with_ocr_gemini(image_file_uploader_object, model_to_use):
    """Extracts text from an image using the provided Gemini multimodal model."""
    model_response_text = ""
    try:
        img_pil = Image.open(image_file_uploader_object)
        prompt_parts = [
            img_pil,
            "\n\nInstruction: Extract all text visible in this image. Present the extracted text clearly. If there are multiple distinct blocks of text, try to preserve their separation if meaningful. Provide only the transcribed text."
        ]
        
        # Using the global llm_model for a direct generation call for OCR
        response = model_to_use.generate_content(prompt_parts)

        if hasattr(response, 'text') and response.text:
            model_response_text = response.text
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
            model_response_text = f"[OCR via Vision Model Blocked: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason}]"
        else:
            model_response_text = "[OCR via Vision Model: No text extracted or unable to process.]"
            
    except Exception as e:
        st.error(f"Vision Model OCR Error: {e}")
        model_response_text = f"Vision Model OCR Error: {e}"
    return model_response_text

# --- UI Layout ---
st.markdown("""
<div style='display: flex; align-items: center; margin-bottom: 1em;'>
    <img src='https://raw.githubusercontent.com/Harivora/my-ai-assistant/main/images/AU%20logo.png' alt='Logo' width='60' style='border-radius: 50%; box-shadow: 0 2px 8px rgba(0,0,0,0.15); margin-right: 18px;'>
    <div>
        <h1 style='margin: 0; font-size: 2.2rem;'>My AI Assistant ✨</h1>
        <div style='color: #aaa; font-size: 1.1rem;'>A versatile AI tool for chat, summarization, vision, OCR, and audio transcription.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar for file uploads and options
with st.sidebar:
    st.header("🛠️ Tools & Uploads")

    # PDF Processing
    pdf_uploader = st.file_uploader("📄 Summarize PDF", type="pdf")
    if pdf_uploader:
        if st.button("Process & Summarize PDF"):
            with st.spinner("📄 Extracting and summarizing PDF..."):
                pdf_text = extract_text_from_pdf(pdf_uploader)
                model_response_text = ""
                if pdf_text:
                    st.session_state.chat_history.append({"role": "user", "parts": [f"Summarize the content of the PDF: {pdf_uploader.name} (approx. {len(pdf_text)} characters extracted)."]})
                    try:
                        # Summarize using the chat session
                        # Ensure the prompt is clear and text is not excessively long for a single API call
                        # Gemini 1.5 Flash has a large context window, but be mindful of limits/costs for very large PDFs.
                        # For extremely long texts, consider chunking and summarizing iteratively.
                        summarization_prompt = f"Please provide a concise summary of the following text extracted from a PDF:\n\n{pdf_text[:1000000]}" # Limit context for safety
                        response = st.session_state.chat_session.send_message(summarization_prompt)
                        if hasattr(response, 'text') and response.text:
                            model_response_text = response.text
                        elif response.prompt_feedback and response.prompt_feedback.block_reason:
                             model_response_text = f"[Summarization Blocked: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason}]"
                        else:
                            model_response_text = "[Could not get summary from model.]"
                    except Exception as e:
                        st.error(f"Error summarizing PDF with AI: {e}")
                        model_response_text = f"Error summarizing PDF: {e}"
                else:
                    st.error("Failed to extract text from PDF. Cannot summarize.")
                    model_response_text = "Failed to extract text from PDF."
                
                if model_response_text: # Even if it's an error message from extraction
                     st.session_state.chat_history.append({"role": "model", "parts": [model_response_text]})
            st.rerun()

    st.markdown("---")
    # Image Description
    image_uploader = st.file_uploader("🖼️ Describe or Query Image", type=["png", "jpg", "jpeg", "webp"])
    if image_uploader:
        st.image(image_uploader, caption="Uploaded Image")
        if st.button("Process Image (Describe)"):
            with st.spinner("🖼️ Analyzing image..."):
                img_pil = Image.open(image_uploader)
                user_image_message_parts = [img_pil, "Describe this image in detail, please."]
                st.session_state.chat_history.append({"role": "user", "parts": user_image_message_parts})
                
                model_response_text = ""
                try:
                    response_stream = st.session_state.chat_session.send_message(user_image_message_parts, stream=True)
                    for chunk in response_stream:
                        if hasattr(chunk, 'text') and chunk.text:
                            model_response_text += chunk.text
                    
                    if not model_response_text: # Handle non-text or blocked responses after stream
                        candidate = response_stream.candidates[0] if response_stream.candidates else None
                        prompt_feedback = response_stream.prompt_feedback
                        if candidate and candidate.finish_reason == 'MAX_TOKENS':
                            model_response_text = "[Response truncated due to maximum token limit]"
                        elif prompt_feedback and prompt_feedback.block_reason:
                            model_response_text = f"[Content blocked: {prompt_feedback.block_reason_message or prompt_feedback.block_reason}]"
                        else:
                            model_response_text = "[The model did not provide a text description for the image.]"
                except Exception as e:
                    st.error(f"Error describing image with AI: {e}")
                    model_response_text = f"Sorry, I encountered an error describing the image: {e}"
                
                st.session_state.chat_history.append({"role": "model", "parts": [model_response_text]})
            st.rerun()

    st.markdown("---")
    # Audio Transcription
    audio_uploader = st.file_uploader("🎤 Transcribe Audio", type=["wav", "mp3", "m4a", "ogg", "flac"])
    if audio_uploader:
        if st.button("Transcribe Audio File"):
            with st.spinner("🎤 Transcribing audio..."):
                transcription_result = transcribe_audio(audio_uploader)
            # Add user request and model response (transcription or error) to history
            st.session_state.chat_history.append({"role": "user", "parts": [f"Transcribe the uploaded audio file: {audio_uploader.name}"]})
            st.session_state.chat_history.append({"role": "model", "parts": [f"{transcription_result}"]}) # Result could be transcription or error string
            st.rerun()

    st.markdown("---")
    # OCR with Vision AI
    ocr_image_uploader = st.file_uploader("👁️ Extract Text from Image (Vision AI OCR)", type=["png", "jpg", "jpeg", "webp"])
    if ocr_image_uploader:
        st.image(ocr_image_uploader, caption="Image for Vision AI OCR")
        if st.button("Extract Text (Vision AI)"):
            with st.spinner("👁️ Performing Vision AI OCR..."):
                extracted_ocr_text = extract_text_with_ocr_gemini(ocr_image_uploader, llm_model) # Pass global model
            
            st.session_state.chat_history.append({"role": "user", "parts": [f"Extract text from this image using Vision AI: {ocr_image_uploader.name}"]})
            st.session_state.chat_history.append({"role": "model", "parts": [f"Extracted Text (Vision AI OCR):\n{extracted_ocr_text}"]})
            st.rerun()
    
    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.chat_session = llm_model.start_chat(history=[]) # Reset model's internal history
        if os.path.exists(MEMORY_FILE):
            os.remove(MEMORY_FILE)
        st.rerun()

# Main Chat Interface
st.header("💬 Chat Window")

# Display chat history
for i, message in enumerate(st.session_state.chat_history):
    role = "assistant" if message["role"] == "model" else message["role"]
    with st.chat_message(role):
        for part in message["parts"]:
            if isinstance(part, Image.Image): 
                st.image(part, width=300)
            elif isinstance(part, str) : 
                 st.markdown(part)
            else:
                st.write(part)

# User input from st.chat_input
user_prompt_text_input = st.chat_input("Ask anything, or describe an uploaded file...")

if user_prompt_text_input:
    # Add user's text message to history
    st.session_state.chat_history.append({"role": "user", "parts": [user_prompt_text_input]})
    save_chat_history()

    # Get model's response (assistant)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response_text = ""
        try:
            response_stream = st.session_state.chat_session.send_message(user_prompt_text_input, stream=True)
            for chunk in response_stream:
                if hasattr(chunk, 'text') and chunk.text:
                    full_response_text += chunk.text
                    message_placeholder.markdown(full_response_text + "▌") # Simulate typing cursor
            message_placeholder.markdown(full_response_text) # Display full response
            
            if not full_response_text: # Handle non-text or blocked responses after stream
                candidate = response_stream.candidates[0] if response_stream.candidates else None
                prompt_feedback = response_stream.prompt_feedback
                if candidate and candidate.finish_reason == 'MAX_TOKENS':
                    full_response_text = "[Response truncated due to maximum token limit]"
                elif prompt_feedback and prompt_feedback.block_reason:
                    full_response_text = f"[Content blocked: {prompt_feedback.block_reason_message or prompt_feedback.block_reason}]"
                else:
                    full_response_text = "[The model did not provide a text response.]"
                message_placeholder.markdown(full_response_text)

        except Exception as e:
            st.error(f"An error occurred with the AI: {e}")
            full_response_text = f"Sorry, I encountered an error: {e}"
            message_placeholder.error(full_response_text)
    
    # Add model's response to history
    st.session_state.chat_history.append({"role": "model", "parts": [full_response_text]})
    save_chat_history()
    # If an explicit rerun is needed after adding model response, uncomment below,
    # but test for duplicate messages or undesired behavior.
    # st.rerun()

# --- Footer ---
# Footer with logo and Gemini API key link (centered)
st.markdown("""
---
<div style='text-align: center; color: #888; font-size: 1em;'>
    Developed by Harikrishna Vora<br>
    <div style='display: flex; justify-content: center; align-items: center; margin-top: 0.5em;'>
        <img src='https://raw.githubusercontent.com/<your-github-username>/<your-repo-name>/main/ChatGPT/images/AU%20logo.png' alt='Logo' width='80' style='border-radius: 50%; box-shadow: 0 2px 8px rgba(0,0,0,0.15);'>
    </div>
    <div style='margin-top: 1em;'>
        <a href='https://aistudio.google.com/app/apikey' target='_blank' style='color: #4f8cff; font-weight: 500; text-decoration: underline;'>Get your Gemini API key here</a>
    </div>
</div>
""", unsafe_allow_html=True)

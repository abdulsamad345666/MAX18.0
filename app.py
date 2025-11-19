# app.py
import streamlit as st
from streamlit_chat import message
import os
import json
import time
from datetime import datetime
import base64
from io import BytesIO

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="Universal AI Assistant v2.0",
    page_icon="🦸‍♂️",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourname',
        'Report a bug': "https://github.com/yourname/issues",
        'About': "# Universal AI Assistant v2.0\nMade with ❤️ in India"
    }
)

# Custom CSS - Desi Premium Look
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px;}
    .stApp > header {background-color: transparent;}
    .avatar-user {background-color: #FF6B6B; border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;}
    .avatar-bot {background-color: #4ECDC4; border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;}
    .chat-message {padding: 1rem; border-radius: 15px; margin: 10px 0; max-width: 80%;}
    .user-msg {background: #FF6B6B; color: white; margin-left: auto;}
    .bot-msg {background: #4ECDC4; color: white;}
    .title {font-size: 3rem; font-weight: bold; text-align: center; color: white; text-shadow: 2px 2px 10px rgba(0,0,0,0.5);}
    .subtitle {text-align: center; color: #f0f0f0; font-size: 1.3rem;}
</style>
""", unsafe_allow_html=True)

# ------------------ IMPORTS ------------------
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.markdown("<h1 style='color:#FF6B6B;'>🦸‍♂️ Universal AI</h1>", unsafe_allow_html=True)
    st.markdown("### Select Model")
    model_choice = st.selectbox("AI Engine", [
        "Grok (xAI)", "Gemini Pro", "Claude 3", "Ollama (Llama3)", "DeepSeek", "OpenAI GPT-4o"
    ])

    st.markdown("### Features")
    if st.checkbox("Enable Voice Input"):
        voice_mode = True
    else:
        voice_mode = False

    if st.checkbox("Enable Memory (RAG)"):
        rag_mode = True
    else:
        rag_mode = False

    st.markdown("---")
    st.caption("Made with 🔥 by Indian Developer")

# ------------------ INITIALIZE SESSION STATE ------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Namaste! Main hoon aapka **Universal AI Assistant v2.0** 🚀\nKya madad chahiye aaj?"}
    ]

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ------------------ FUNCTIONS ------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Boliye bhai...")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio, language="hi-IN")
            return text
        except:
            return "Sorry, nahi samjha"

def search_web(query):
    url = f"https://www.google.com/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    snippets = []
    for g in soup.find_all('div', class_='g'):
        text = g.get_text()
        if len(text) < 300:
            snippets.append(text)
    return "\n\n".join(snippets[:3]) if snippets else "Kuch nahi mila bhai"

def process_uploaded_file(uploaded_file):
    file_path = f"./uploads/{uploaded_file.name}"
    os.makedirs("./uploads", exist_ok=True)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load based on type
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif uploaded_file.name.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(documents=splits, embedding=get_embeddings())
    st.session_state.vectorstore = vectorstore
    st.success(f"File {uploaded_file.name} loaded in memory! Ab file ke baare mein pooch sakte ho")

def get_response(prompt):
    # RAG if enabled and files uploaded
    if rag_mode and st.session_state.vectorstore:
        retriever = st.session_state.vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-3.5-turbo"),  # fallback
            chain_type="stuff",
            retriever=retriever
        )
        try:
            return qa_chain.run(prompt)
        except:
            pass

    # Model-specific responses
    if "Grok" in model_choice:
        return f"[GROK MODE] {prompt} → Ye toh bahut badhiya sawal hai bhai! Grok bolta hai: Abhi xAI API nahi hai public, toh main pretend kar raha hoon 😂"
    
    elif "Gemini" in model_choice:
        try:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(f"In Hinglish: {prompt}")
            return response.text
        except:
            return "Gemini API key nahi hai bhai!"

    elif "Ollama" in model_choice:
        try:
            llm = Ollama(model="llama3")
            return llm.invoke(prompt)
        except:
            return "Ollama server nahi chal raha!"

    else:
        return f"[SIMULATED {model_choice}] {prompt}\n\nBhai, main abhi dummy response de raha hoon kyunki real API key nahi hai. Par asli mein yeh jawdropping hoga! 🚀"

# ------------------ MAIN UI ------------------
st.markdown("<h1 class='title'>Universal AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Har System Mein • Har Kaam Mein • Full Desi Power 🔥</p>", unsafe_allow_html=True)

# Chat History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=str(time.time()) + msg["content"][:10])
    else:
        message(msg["content"], is_user=False, avatar_style="thumbs", key=str(time.time()))

# File Upload for RAG
st.sidebar.file_uploader("Upload File for RAG (PDF/DOCX/TXT)", type=["pdf","docx","txt"], on_change=process_uploaded_file, key="rag_file")

# Input Area
col1, col2, col3 = st.columns([5,1,1])

with col1:
    user_input = st.chat_input("Yahan type karo ya voice use karo...")

with col2:
    if st.button("🎤"):
        if voice_mode:
            text = voice_input()
            user_input = text
            st.write(f"You said: {text}")

with col3:
    if st.button("🌐"):
        if user_input:
            with st.spinner("Internet se la raha hoon..."):
                web_result = search_web(user_input)
                st.session_state.messages.append({"role": "assistant", "content": f"**Web Search Result:**\n\n{web_result}"})
                st.rerun()

# Process Input
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    message(user_input, is_user=True)

    with st.chat_message("assistant"):
        with st.spinner("Jawab bana raha hoon..."):
            time.sleep(1)
            response = get_response(user_input)
            st.markdown(response)
            if voice_mode:
                speak(response.replace("**", ""))

    st.session_state.messages.append({"role": "assistant", "content": response})

# Quick Buttons
st.markdown("### Quick Actions")
c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("Email Likho"):
        st.session_state.messages.append({"role": "user", "content": "Ek professional leave email likh do"})
        st.rerun()
with c2:
    if st.button("Code Banao"):
        st.session_state.messages.append({"role": "role": "user", "content": "Python mein calculator banao"})
        st.rerun()
with c3:
    if st.button("News Batao"):
        st.session_state.messages.append({"role": "user", "content": "Aaj ki top 5 India news"})
        st.rerun()
with c4:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:#aaa;'>Universal AI Assistant v2.0 • Desh Ka Apna Super AI 🧡🤍💚</p>", unsafe_allow_html=True)

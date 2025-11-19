# modules/chat_engine.py
import os
import json
import requests
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from threading import Thread
import time
from datetime import datetime

class AIChatEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.offline_mode = False
        self.model_name = "Qwen/Qwen2.5-7B-Instruct"  # Sabse tez + Hindi mein mast
        # Alternatives: "mistralai/Mistral-7B-Instruct-v0.3", "google/gemma-2-9b-it", "microsoft/DialoGPT-medium"
        
        print(f"Device detected: {self.device.upper()}")
        self.load_best_model()

    def load_best_model(self):
        """Smart model loading: Fast local → Quantized → Tiny fallback"""
        try:
            # Option 1: Try 4-bit quantized 7B model (runs on 8GB VRAM / 16GB RAM)
            if torch.cuda.is_available():
                print("Loading 4-bit quantized 7B model (Best performance)...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    quantization_config=quantization_config,
                    trust_remote_code=True
                )
            else:
                # CPU pe lightweight model
                print("Loading CPU-optimized small model...")
                self.tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
                self.model = AutoModelForCausalLM.from_pretrained(
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    torch_dtype=torch.float16,
                    device_map="cpu"
                )
            print("LOCAL MODEL LOADED SUCCESSFULLY! (Offline Ready)")
            self.offline_mode = True

        except Exception as e:
            print(f"Local model failed → Switching to Online APIs: {e}")
            self.offline_mode = False

    def generate_response(self, prompt, user_name="User", context=""):
        """Main function - returns rich, Hinglish, helpful response"""
        
        full_prompt = f"""You are a super intelligent, funny, helpful Indian AI assistant.
You speak in natural Hinglish (Hindi + English mixed), use emojis, and sound like a dost.
Current time: {datetime.now().strftime('%I:%M %p, %d %b %Y')}

Context: {context}
User: {prompt}
Assistant:"""

        # 1. Try Local Model (Best Quality + Offline)
        if self.offline_mode and self.model:
            try:
                inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=250,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                clean_response = response.split("Assistant:")[-1].strip()
                return self._beautify_response(clean_response)
            except Exception as e:
                print(f"Local model error: {e}")

        # 2. Online Fallbacks (Multiple Free APIs)
        online_response = self._try_online_apis(prompt)
        if online_response:
            return self._beautify_response(online_response)

        # 3. Final Desi Fallback
        return self._desi_fallback(prompt, user_name)

    def _try_online_apis(self, prompt):
        """Multiple free online APIs with auto-fallback"""
        apis = [
            {
                "name": "Groq (Llama3 70B - Blazing Fast)",
                "url": "https://api.groq.com/openai/v1/chat/completions",
                "headers": {"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
                "payload": {
                    "model": "llama3-70b-8192",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.8
                }
            },
            {
                "name": "DeepSeek (Free & Powerful)",
                "url": "https://api.deepseek.com/v1/chat/completions",
                "headers": {"Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}", "Content-Type": "application/json"},
                "payload": {
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": f"Hinglish mein jawab do: {prompt}"}]
                }
            },
            {
                "name": "HuggingFace Inference",
                "url": "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-14B-Instruct",
                "headers": {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"},
                "payload": {"inputs": prompt, "parameters": {"max_new_tokens": 200}}
            }
        ]

        for api in apis:
            try:
                response = requests.post(api["url"], headers=api["headers"], json=api.get("payload", api["payload"]), timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    if "choices" in data:
                        return data["choices"][0]["message"]["content"]
                    elif isinstance(data, list) and len(data) > 0:
                        return data[0].get("generated_text", "Kuch toh gadbad hai...")
            except:
                continue
        return None

    def _desi_fallback(self, prompt, user_name):
        """When everything fails — Desi bhai never gives up!"""
        prompt = prompt.lower()

        responses = {
            "hi": f"Arre {user_name}! Kya haal hai bhai? 🔥",
            "hello": "Namaste ji! Kya seva kar sakta hoon aaj?",
            "time": f"Abhi time hai {datetime.now().strftime('%I:%M %p')} ⏰",
            "date": f"Aaj hai {datetime.now().strftime('%d %B %Y, %A')} 📅",
            "email": """**Professional Email Template**

Subject: Request for Leave - [Date]

Dear Sir/Madam,

I am writing to request leave from [start date] to [end date] due to [reason].

Thank you for your understanding.

Best regards,  
[Your Name]""",
            "love": "Arre bhai! Pyaar toh banta hai ❤️ Par pehle chai peete hain?",
            "joke": "Ek baar ek banda apni girlfriend se: 'Tum sundar ho'\nWoh boli: 'Thanks'\nBanda: 'Arre maine abhi start hi kiya tha!' 😂",
            "motivation": "Bhai, yaad rakh: \n'Sapne woh nahi jo sote waqt dekhe jaate hain... \nSapne woh hain jo sone na de!' \nAaj ka din tera hai! 🚀"
        }

        for key, resp in responses.items():
            if key in prompt:
                return resp + "\n\n(Internet nahi hai, par dil se dil tak baat pahunch gayi na? 😎)"

        return f"Bhai tune poocha: '{prompt}'\n\nAbhi internet nahi hai, lekin jab aayega na — main tujhe aisa jawab doonga ki tu dang reh jayega! 🔥\n\nTab tak chai pi le ☕"

    def _beautify_response(self, text):
        """Make every response look premium"""
        text = text.replace("**", "").strip()
        if not any(c in text for c in ["?", "!", "😂", "🔥", "🚀", "❤️", "😎"]):
            text += " 🔥"
        return text

    def stream_response(self, prompt):
        """For Streamlit streaming effect (like ChatGPT)"""
        if self.offline_mode and self.model:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            tokens = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.8,
                do_sample=True,
                streamer=True
            )
            for token in tokens:
                yield self.tokenizer.decode(token, skip_special_tokens=True)
        else:
            response = self.generate_response(prompt)
            for word in response.split():
                yield word + " "
                time.sleep(0.03)

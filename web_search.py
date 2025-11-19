# modules/web_search.py
import requests
from bs4 import BeautifulSoup
import json
import time
import random
from typing import List, Dict, Any
import os
from datetime import datetime

class WebSearch:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': random.choice([
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
            ])
        })
        # Free APIs (working 2025)
        self.apis = {
            "serper": "https://google.serper.dev/search",      # Best free tier
            "brave": "https://api.search.brave.com/res/v1/web/search",
            "jina": "https://s.jina.ai/",                      # Free AI search
            "duckduckgo": "https://html.duckduckgo.com/html/",
        }

    def search(self, query: str, num_results: int = 10) -> List[Dict]:
        """ULTIMATE SEARCH - Tries 5 engines → Returns best results"""
        query = query.strip()
        if not query:
            return [{"title": "Query empty hai bhai!", "link": "", "snippet": "Kuch toh likh..."}]

        # 1. Serper.dev (Best free Google results)
        results = self._serper_search(query, num_results)
        if results and len(results) > 3:
            return self._clean_results(results)

        # 2. Brave Search API
        results = self._brave_search(query)
        if results:
            return self._clean_results(results)

        # 3. Jina AI Search (Free + AI summaries)
        results = self._jina_search(query)
        if results:
            return results

        # 4. Direct Scraping (Google + DuckDuckGo)
        results = self._scrape_google(query) or self._scrape_duckduckgo(query)
        if results:
            return self._clean_results(results)

        # 5. Final Fallback
        return [{
            "title": "Internet nahi chal raha bhai!",
            "link": "https://www.google.com",
            "snippet": f"Query: {query}\nJab net aayega tab bata denge! Abhi chai pi lo ☕"
        }]

    def _serper_search(self, query, num=10):
        """Best free Google results (100 searches/day free)"""
        try:
            payload = {"q": query, "num": num}
            headers = {"X-API-KEY": os.getenv("SERPER_API_KEY", ""), "Content-Type": "application/json"}
            if not headers["X-API-KEY"]:
                return None
            response = self.session.post(self.apis["serper"], json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = []
                for item in data.get("organic", [])[:num]:
                    results.append({
                        "title": item.get("title", "No title"),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", "No description")
                    })
                return results
        except:
            return None
        return None

    def _brave_search(self, query):
        """Brave Search - Free tier available"""
        try:
            headers = {"X-Subscription-Token": os.getenv("BRAVE_API_KEY", "")}
            if not headers.get("X-Subscription-Token"):
                return None
            params = {"q": query, "count": 10}
            response = self.session.get(self.apis["brave"], headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = []
                for item in data.get("web", {}).get("results", []):
                    results.append({
                        "title": item.get("title"),
                        "link": item.get("url"),
                        "snippet": item.get("description")
                    })
                return results
        except:
            pass
        return None

    def _jina_search(self, query):
        """Jina.ai - Free AI-powered search with summary"""
        try:
            url = f"{self.apis['jina']}{query}"
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                text = response.text
                # Jina returns markdown
                lines = text.split("\n")
                results = []
                current = {}
                for line in lines:
                    if line.startswith("**") and "](http" in line:
                        if current:
                            results.append(current)
                        title = line.split("**")[1].split("](")[0]
                        link = line.split("](")[1].split(")")[0]
                        current = {"title": title, "link": link, "snippet": ""}
                    elif line.strip() and current:
                        current["snippet"] += line + " "
                if current:
                    results.append(current)
                return results[:8]
        except:
            pass
        return None

    def _scrape_google(self, query):
        """Direct Google scrape (with anti-blocking)"""
        try:
            url = f"https://www.google.com/search?q={query}&num=10&hl=en"
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            for g in soup.find_all('div', class_='g')[:8]:
                a = g.find('a')
                h3 = g.find('h3')
                if a and h3:
                    link = a['href']
                    if link.startswith('/url?q='):
                        link = link.split('/url?q=')[1].split('&')[0]
                    results.append({
                        "title": h3.text,
                        "link": link,
                        "snippet": g.find('span', class_='aCOpRe') or "No description"
                    })
            return results
        except:
            return None

    def _scrape_duckduckgo(self, query):
        try:
            response = self.session.post(self.apis["duckduckgo"], data={"q": query})
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            for result in soup.find_all('a', class_='result__a', limit=10):
                title = result.get_text()
                link = result.get('href')
                if "uddg=" in link:
                    link = link.split("uddg=")[1].split("&")[0]
                snippet = result.find_next('a', class_='result__snippet')
                results.append({
                    "title": title,
                    "link": link,
                    "snippet": snippet.text if snippet else ""
                })
            return results
        except:
            return None

    def get_news(self, topic: str = "India", num: int = 8):
        """Latest News with AI summary"""
        results = self.search(f"{topic} latest news site:ndtv.com | site:indianexpress.com | site:timesofindia.com")
        news = []
        for r in results[:num]:
            news.append(f"**{r['title']}**\n{r.get('snippet', '')[:200]}...\n[Read More]({r['link']})\n")
        return "\n\n".join(news) if news else "Aaj koi khabar nahi mili bhai!"

    def youtube_search(self, query: str):
        """Search YouTube videos"""
        return self.search(f"{query} site:youtube.com")

    def price_check(self, product: str):
        """Check price on Amazon/Flipkart"""
        return self.search(f"{product} price site:amazon.in OR site:flipkart.com")

    def _clean_results(self, results):
        clean = []
        seen = set()
        for r in results:
            link = r.get("link", "")
            if link and link not in seen and "http" in link:
                seen.add(link)
                clean.append({
                    "title": r.get("title", "No Title").replace(" - YouTube", ""),
                    "link": link,
                    "snippet": r.get("snippet", "No description available")[:300]
                })
        return clean if clean else [{"title": "Kuch nahi mila!", "link": "", "snippet": "Try different keywords..."}]

    def quick_answer(self, question: str):
        """For factual questions - uses Jina + Serper"""
        result = self._jina_search(question + " answer")
        if result and len(result) > 0:
            return f"**Jawab:** {result[0].get('snippet', '')[:500]}"
        return "Exact jawab nahi mila, par upar search results dekh lo!"

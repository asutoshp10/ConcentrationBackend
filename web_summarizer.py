import google.generativeai as genai
import requests
from bs4 import BeautifulSoup

genai.configure(api_key="AIzaSyCt07QcDpiIcmMafQ8EzC1U0fhF2apfZ8o")
model = genai.GenerativeModel("models/gemini-1.5-flash")

def fetch_website_content(url):
    """Fetch and extract text content from a website"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    except requests.RequestException as e:
        print(f"Error fetching website: {e}")
        return None
    except Exception as e:
        print(f"Error parsing content: {e}")
        return None

def summarize_web(link):
    """Summarize website content using Gemini AI"""
    # First, fetch the website content
    content = fetch_website_content(link)
    
    if not content:
        return "Error: Could not fetch website content"
    
    max_chars = 30000
    if len(content) > max_chars:
        content = content[:max_chars] + "..."
    
    prompt_s = f"""
    Please provide a detailed summary of the following website content:
    
    Website URL: {link}
    
    Content:
    {content}
    
    Please include:
    1. Main topic and key points
    2. Important details and facts
    3. Conclusions or takeaways
    """
    
    try:
        response = model.generate_content(prompt_s)
        return response.text
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "Error: Could not generate summary"

# Example usage
if __name__ == "__main__":
    url = "https://example.com"
    summary = summarize_web(url)
    print(summary)

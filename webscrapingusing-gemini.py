"""
AI-Powered Web Scraper using Google Gemini
Uses Google Gemini AI to intelligently scrape and crawl websites.

Features:
- AI-powered content extraction and summarization
- Intelligent crawling decisions
- Structured data extraction using AI
- Natural language queries
"""

import os
import json
import time
import re
from typing import Dict, List, Optional, Any, Tuple
import requests
from bs4 import BeautifulSoup

# Advanced scraping tools (optional imports)
try:
    from playwright.sync_api import sync_playwright, Browser, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    # Try loading from parent directory (project root) and current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    load_dotenv(os.path.join(parent_dir, '.env'))  # Try parent directory first
    load_dotenv()  # Then try current directory
except ImportError:
    # python-dotenv not installed, continue without it
    pass
except Exception:
    # If loading fails, continue without .env
    pass


# ============================================================================
# BLOCK DETECTION AND CLASSIFICATION
# ============================================================================

class BlockDetector:
    """Detects and classifies different types of website blocks."""
    
    @staticmethod
    def detect_block_type(response: requests.Response, html_content: str = "") -> str:
        """
        Detect and classify the type of block.
        
        Returns:
            'cloudflare' - Cloudflare protection
            '403' - Simple 403 Forbidden
            'javascript' - Requires JavaScript
            'captcha' - CAPTCHA challenge
            'rate_limit' - Rate limiting
            'none' - No block detected
        """
        status_code = response.status_code
        
        # Check status code
        if status_code == 403:
            html_lower = html_content.lower()
            
            # Cloudflare detection
            if any(indicator in html_lower for indicator in [
                'cloudflare', 'cf-ray', 'checking your browser', 
                'ddos protection', 'just a moment', 'please wait'
            ]):
                return 'cloudflare'
            
            # CAPTCHA detection
            if any(indicator in html_lower for indicator in [
                'captcha', 'recaptcha', 'hcaptcha', 'verify you are human'
            ]):
                return 'captcha'
            
            return '403'
        
        if status_code == 429:
            return 'rate_limit'
        
        # Check for JavaScript requirement
        html_lower = html_content.lower()
        if any(indicator in html_lower for indicator in [
            'javascript is required', 'enable javascript', 
            'noscript', 'please enable javascript'
        ]):
            return 'javascript'
        
        return 'none'
    
    @staticmethod
    def can_bypass_with_playwright(block_type: str) -> bool:
        """Check if Playwright can bypass this block type."""
        return block_type in ['cloudflare', 'javascript', '403']
    
    @staticmethod
    def can_bypass_with_selenium(block_type: str) -> bool:
        """Check if Selenium can bypass this block type."""
        return block_type in ['javascript', '403']


# ============================================================================
# AI CLIENT SETUP - Gemini AI
# ============================================================================

class AIScraper:
    """AI-powered web scraper using Google Gemini AI."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "models/gemini-2.5-flash"):
        """
        Initialize AI scraper with Gemini.
        
        Args:
            api_key: Google API key (or set GOOGLE_API_KEY environment variable)
            model: Gemini model to use (models/gemini-2.5-flash, models/gemini-pro-latest, models/gemini-2.5-pro)
        """
        try:
            import google.generativeai as genai
            
            self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
            
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model)
            self.model_name = model
            
        except ImportError:
            raise ImportError("google-generativeai package not installed. Install with: pip install google-generativeai")
    
    def _fetch_with_requests(self, url: str) -> Tuple[Optional[requests.Response], Optional[str], Dict[str, Any]]:
        """Try fetching with requests library. Returns (response, html_content, error_dict)."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
        session = requests.Session()
        session.headers.update(headers)
        try:
            response = session.get(url, timeout=15, allow_redirects=True)
            return response, response.text, {}
        except Exception as e:
            return None, None, {"error": str(e), "method": "requests"}
    
    def _fetch_with_playwright(self, url: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """Fetch using Playwright (bypasses JavaScript and many blocks)."""
        if not PLAYWRIGHT_AVAILABLE:
            return None, {"error": "Playwright not installed. Install with: pip install playwright && playwright install", "method": "playwright"}
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=['--disable-blink-features=AutomationControlled'])
                context = browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    locale='en-US', timezone_id='America/New_York'
                )
                page = context.new_page()
                page.goto(url, wait_until='networkidle', timeout=30000)
                time.sleep(2)
                html_content = page.content()
                browser.close()
                return html_content, {}
        except Exception as e:
            return None, {"error": str(e), "method": "playwright"}
    
    def _fetch_with_selenium(self, url: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """Fetch using Selenium (fallback method)."""
        if not SELENIUM_AVAILABLE:
            return None, {"error": "Selenium not installed. Install with: pip install selenium", "method": "selenium"}
        driver = None
        try:
            chrome_options = ChromeOptions()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(30)
            driver.get(url)
            time.sleep(3)
            html_content = driver.page_source
            driver.quit()
            return html_content, {}
        except Exception as e:
            if driver:
                try:
                    driver.quit()
                except:
                    pass
            return None, {"error": str(e), "method": "selenium"}
    
    def _fetch_webpage(self, url: str) -> Dict[str, Any]:
        """
        Smart webpage fetcher that tries multiple methods to bypass blocks.
        Strategy: 1) Try requests, 2) Detect block type, 3) Try Playwright, 4) Fallback to Selenium.
        """
        # Step 1: Try requests first
        response, html_content, error = self._fetch_with_requests(url)
        
        if response and response.status_code == 200:
            soup = BeautifulSoup(html_content, 'html.parser')
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()
            text_content = soup.get_text(separator='\n', strip=True)
            html_limited = str(soup)[:50000]
            return {
                "url": url, "status": 200, "text": text_content, "html": html_limited,
                "title": soup.title.string if soup.title else None,
                "links": [a.get('href') for a in soup.find_all('a', href=True)],
                "error": None, "method": "requests"
            }
        
        # Step 2: Detect block type
        block_type = 'unknown'
        if response:
            block_type = BlockDetector.detect_block_type(response, html_content or "")
            print(f"⚠️  Block detected: {block_type} (Status: {response.status_code})")
        
        # Step 3: Try Playwright
        if BlockDetector.can_bypass_with_playwright(block_type) or not response:
            print("🔄 Trying Playwright...")
            html_content, error = self._fetch_with_playwright(url)
            if html_content and not error:
                soup = BeautifulSoup(html_content, 'html.parser')
                for element in soup(["script", "style", "nav", "footer", "header"]):
                    element.decompose()
                text_content = soup.get_text(separator='\n', strip=True)
                html_limited = str(soup)[:50000]
                return {
                    "url": url, "status": 200, "text": text_content, "html": html_limited,
                    "title": soup.title.string if soup.title else None,
                    "links": [a.get('href') for a in soup.find_all('a', href=True)],
                    "error": None, "method": "playwright", "block_bypassed": block_type
                }
        
        # Step 4: Fallback to Selenium
        if BlockDetector.can_bypass_with_selenium(block_type) or not response:
            print("🔄 Trying Selenium...")
            html_content, error = self._fetch_with_selenium(url)
            if html_content and not error:
                soup = BeautifulSoup(html_content, 'html.parser')
                for element in soup(["script", "style", "nav", "footer", "header"]):
                    element.decompose()
                text_content = soup.get_text(separator='\n', strip=True)
                html_limited = str(soup)[:50000]
                return {
                    "url": url, "status": 200, "text": text_content, "html": html_limited,
                    "title": soup.title.string if soup.title else None,
                    "links": [a.get('href') for a in soup.find_all('a', href=True)],
                    "error": None, "method": "selenium", "block_bypassed": block_type
                }
        
        # All methods failed
        error_msg = error.get("error", "All scraping methods failed")
        if response:
            error_msg = f"Block type: {block_type}, Status: {response.status_code}, {error_msg}"
        return {
            "url": url, "status": response.status_code if response else None,
            "text": None, "html": None, "error": error_msg,
            "block_type": block_type, "methods_tried": ["requests", "playwright", "selenium"]
        }
    
    def scrape_with_ai(self, url: str, extraction_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Scrape a webpage and use AI to extract and summarize content.
        
        Args:
            url: URL to scrape
            extraction_prompt: Custom prompt for what to extract (optional)
        
        Returns:
            Dictionary with AI-processed content
        """
        # Fetch webpage
        webpage = self._fetch_webpage(url)
        
        if webpage.get("error"):
            return webpage
        
        # Prepare AI prompt
        if extraction_prompt:
            prompt = f"""You are an expert web scraper. Analyze the following webpage content and extract information based on this request:

{extraction_prompt}

Webpage URL: {url}
Webpage Title: {webpage.get('title', 'N/A')}

Webpage Content:
{webpage['text'][:15000]}

Please provide:
1. A clear summary of the main content
2. Key information extracted based on the request
3. Any important details, links, or data points
4. Structured data if applicable (JSON format)

Format your response clearly and comprehensively."""
        else:
            prompt = f"""You are an expert web scraper. Analyze the following webpage and provide a comprehensive summary.

Webpage URL: {url}
Webpage Title: {webpage.get('title', 'N/A')}

Webpage Content:
{webpage['text'][:15000]}

Please provide:
1. A clear summary of what this webpage is about
2. Main topics and key information
3. Important links and resources mentioned
4. Any structured data or facts that stand out

Format your response clearly."""
        
        try:
            # Call Gemini API
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 2000,
                }
            )
            
            ai_summary = response.text
            
            return {
                "url": url,
                "status": webpage["status"],
                "title": webpage.get("title"),
                "raw_content": webpage["text"][:1000] + "..." if len(webpage["text"]) > 1000 else webpage["text"],
                "ai_summary": ai_summary,
                "links": webpage.get("links", [])[:20],  # Limit links
                "error": None
            }
        
        except Exception as e:
            return {
                "url": url,
                "error": f"AI processing failed: {str(e)}",
                "raw_content": webpage.get("text", "")[:500]
            }
    
    def crawl_with_ai(self, start_url: str, max_pages: int = 5, crawl_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Intelligently crawl a website using AI to decide which pages to visit.
        
        Args:
            start_url: Starting URL
            max_pages: Maximum number of pages to crawl
            crawl_prompt: What kind of content to look for (optional)
        
        Returns:
            List of dictionaries with AI-processed content from crawled pages
        """
        visited = set()
        to_visit = [start_url]
        results = []
        
        while to_visit and len(results) < max_pages:
            current_url = to_visit.pop(0)
            
            if current_url in visited:
                continue
            
            visited.add(current_url)
            
            print(f"Crawling: {current_url} ({len(results) + 1}/{max_pages})")
            
            # Scrape current page
            result = self.scrape_with_ai(current_url, crawl_prompt)
            results.append(result)
            
            if result.get("error"):
                continue
            
            # Use AI to decide which links to follow
            if len(results) < max_pages and result.get("links"):
                links = result["links"]
                
                # Filter and normalize links
                base_domain = "/".join(start_url.split("/")[:3])
                relevant_links = []
                
                for link in links[:10]:  # Check first 10 links
                    if not link:
                        continue
                    
                    # Convert relative to absolute URLs
                    if link.startswith("/"):
                        full_url = base_domain + link
                    elif link.startswith("http"):
                        full_url = link
                    else:
                        continue
                    
                    # Only follow links from same domain
                    if base_domain in full_url and full_url not in visited:
                        relevant_links.append(full_url)
                
                # Use AI to prioritize which links to crawl next
                if relevant_links and len(results) < max_pages:
                    links_text = "\n".join([f"- {link}" for link in relevant_links[:5]])
                    
                    ai_prompt = f"""You are analyzing a website crawl. Based on the current page content and the goal:
                    
{crawl_prompt or "Find relevant and important pages"}

Here are potential links to visit next:
{links_text}

Which links should be prioritized? Respond with just the URLs (one per line) that seem most relevant. If none are relevant, respond with "NONE"."""
                    
                    try:
                        ai_response = self.model.generate_content(
                            ai_prompt,
                            generation_config={
                                "temperature": 0.2,
                                "max_output_tokens": 500,
                            }
                        )
                        
                        ai_links = ai_response.text.strip()
                        
                        if ai_links.upper() != "NONE":
                            for line in ai_links.split("\n"):
                                url = line.strip().lstrip("- ").strip()
                                if url.startswith("http") and url not in visited and url not in to_visit:
                                    to_visit.append(url)
                                    if len(to_visit) >= 3:  # Limit queue size
                                        break
                    
                    except Exception as e:
                        # Fallback: add first few links if AI fails
                        for link in relevant_links[:2]:
                            if link not in visited and link not in to_visit:
                                to_visit.append(link)
        
        return results
    
    def extract_structured_data(self, url: str, data_schema: str) -> Dict[str, Any]:
        """
        Extract structured data from a webpage using AI based on a schema.
        
        Args:
            url: URL to scrape
            data_schema: Description of what structured data to extract
        
        Returns:
            Dictionary with extracted structured data
        """
        webpage = self._fetch_webpage(url)
        
        if webpage.get("error"):
            return webpage
        
        prompt = f"""Extract structured data from this webpage based on the following schema:

Schema: {data_schema}

Webpage URL: {url}
Webpage Content:
{webpage['text'][:15000]}

Extract the requested data and return it as a JSON object. Only include fields that are actually present on the page. If a field is not found, use null. Return ONLY valid JSON, no additional text."""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 2000,
                }
            )
            
            # Try to extract JSON from response
            response_text = response.text.strip()
            
            # Try to find JSON in the response (sometimes Gemini adds extra text)
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            extracted_data = json.loads(response_text)
            
            return {
                "url": url,
                "status": webpage["status"],
                "extracted_data": extracted_data,
                "error": None
            }
        
        except json.JSONDecodeError as e:
            return {
                "url": url,
                "error": f"Failed to parse JSON from AI response: {str(e)}. Response: {response.text[:200]}"
            }
        except Exception as e:
            return {
                "url": url,
                "error": f"Data extraction failed: {str(e)}"
            }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def scrape_with_ai(url: str, api_key: Optional[str] = None, extraction_prompt: Optional[str] = None, model: str = "models/gemini-2.5-flash") -> Dict[str, Any]:
    """
    Quick function to scrape a webpage with AI.
    
    Args:
        url: URL to scrape
        api_key: Google API key (optional, can use GOOGLE_API_KEY env var)
        extraction_prompt: What to extract (optional)
        model: Gemini model to use
    
    Returns:
        Dictionary with AI-processed content
    """
    scraper = AIScraper(api_key=api_key, model=model)
    return scraper.scrape_with_ai(url, extraction_prompt)


def crawl_with_ai(start_url: str, max_pages: int = 5, api_key: Optional[str] = None, crawl_prompt: Optional[str] = None, model: str = "models/gemini-2.5-flash") -> List[Dict[str, Any]]:
    """
    Quick function to crawl a website with AI.
    
    Args:
        start_url: Starting URL
        max_pages: Maximum pages to crawl
        api_key: Google API key (optional)
        crawl_prompt: What content to look for
        model: Gemini model to use
    
    Returns:
        List of AI-processed pages
    """
    scraper = AIScraper(api_key=api_key, model=model)
    return scraper.crawl_with_ai(start_url, max_pages, crawl_prompt)


def extract_data(url: str, schema: str, api_key: Optional[str] = None, model: str = "models/gemini-2.5-flash") -> Dict[str, Any]:
    """
    Quick function to extract structured data with AI.
    
    Args:
        url: URL to scrape
        schema: Description of data to extract
        api_key: Google API key (optional)
        model: Gemini model to use
    
    Returns:
        Dictionary with extracted structured data
    """
    scraper = AIScraper(api_key=api_key, model=model)
    return scraper.extract_structured_data(url, schema)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_results(data: Any, filename: str):
    """Save scraping results to JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {filename}")


def print_summary(result: Dict[str, Any]):
    """Print a summary of scraping results."""
    print("\n" + "="*70)
    print("AI SCRAPING RESULTS (Gemini)")
    print("="*70)
    print(f"URL: {result.get('url', 'N/A')}")
    print(f"Status: {result.get('status', 'N/A')}")
    
    if result.get('error'):
        print(f"Error: {result['error']}")
    else:
        if result.get('ai_summary'):
            print(f"\nAI Summary:\n{result['ai_summary']}")
        
        if result.get('extracted_data'):
            print(f"\nExtracted Data:\n{json.dumps(result['extracted_data'], indent=2)}")
        
        if result.get('links'):
            print(f"\nFound {len(result['links'])} links")
    
    print("="*70 + "\n")


# ============================================================================
# MAIN USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("GOOGLE_API_KEY not set!")
        exit(1)
    
    # Get user input
    print("=" * 70)
    print("AI-Powered Web Scraper (Google Gemini)")
    print("=" * 70)
    start_url = input("\nEnter the website URL: ").strip()
    
    if not start_url:
        print("Error: URL is required!")
        exit(1)
    
    # Ensure URL has protocol
    if not start_url.startswith(("http://", "https://")):
        start_url = "https://" + start_url

    # Ask crawling decision
    crawl_prompt = input(
        "\nWhat kind of pages should the AI look for when crawling?\n"
        "(example: documentation, pricing, job listings, blog posts):\n> "
    ).strip() or "Find relevant and important pages"

    scrape_prompt = input(
        "\nWhat kind of data should the AI extract from each page?\n"
        "(example: product names and prices, contact information, article content):\n> "
    ).strip() or "Extract key information and summarize the content"

    max_pages = int(input("\nHow many pages to crawl? (default 3): ") or 3)

    # Example 1: AI-powered crawling
    print("\n" + "=" * 70)
    print("Example 1: AI-Powered Website Crawling (Gemini)")
    print("=" * 70)
    print("\nStarting crawl...\n")

    results = crawl_with_ai(
        start_url=start_url,
        max_pages=max_pages,
        crawl_prompt=crawl_prompt
    )

    save_results(results, "runtime_crawl_results.json")
    
    # Print summary of crawled pages
    print(f"\n Successfully crawled {len(results)} pages")
    for i, page in enumerate(results, 1):
        if page.get('error'):
            print(f"  Page {i}: Error - {page.get('url', 'N/A')}")
        else:
            print(f"  Page {i}: {page.get('url', 'N/A')}")
            if page.get('ai_summary'):
                summary_preview = page['ai_summary'][:100] + "..." if len(page['ai_summary']) > 100 else page['ai_summary']
                print(f"         Summary: {summary_preview}")
    
    print("\n" + "="*70 + "\n")
    
    # Example 2: AI-powered scraping (single page)
    print("Example 2: AI-Powered Website Scraping (Gemini)")
    print("-" * 70)
    
    print("\nStarting Scrape...\n")

    result = scrape_with_ai(
        url=start_url,
        extraction_prompt=scrape_prompt
    )

    save_results(result, "runtime_scrape_results.json")
    print_summary(result)
    
    print("\n" + "="*70 + "\n")


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
from typing import Dict, List, Optional, Any
import requests
from bs4 import BeautifulSoup

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
    
    def _fetch_webpage(self, url: str) -> Dict[str, Any]:
        """
        Fetch webpage content.
        
        Args:
            url: URL to fetch
        
        Returns:
            Dictionary with HTML content and metadata
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()
            
            # Get clean text
            text_content = soup.get_text(separator='\n', strip=True)
            
            # Get HTML (limited to avoid token limits)
            html_content = str(soup)[:50000]  # Limit HTML to 50k chars
            
            return {
                "url": url,
                "status": response.status_code,
                "text": text_content,
                "html": html_content,
                "title": soup.title.string if soup.title else None,
                "links": [a.get('href') for a in soup.find_all('a', href=True)],
                "error": None
            }
        
        except Exception as e:
            return {
                "url": url,
                "status": None,
                "text": None,
                "html": None,
                "error": str(e)
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
{webpage['text'][:15000]}  # Limit to avoid token limits

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
        print("\nTo set it, choose one of these methods:")
        print("\n1. Create a .env file (recommended):")
        print("   - Create a .env file in the project root")
        print("   - Add your key: GOOGLE_API_KEY=your-key-here")
        print("\n2. Set environment variable:")
        print("   Windows PowerShell: $env:GOOGLE_API_KEY='your-key-here'")
        print("   Windows CMD: set GOOGLE_API_KEY=your-key-here")
        print("   Linux/Mac: export GOOGLE_API_KEY='your-key-here'")
        print("\n3. Set it in Python:")
        print("   import os")
        print("   os.environ['GOOGLE_API_KEY'] = 'your-key-here'")
        print("\nTo get a Google API key:")
        print("   1. Go to https://makersuite.google.com/app/apikey")
        print("   2. Create a new API key")
        print("   3. Copy and use it here")
        print("\nExiting. Please set the API key first.\n")
        exit(1)
    
    # Example 1: Basic AI scraping
    print("Example 1: AI-Powered Web Scraping (Gemini)")
    print("-" * 70)
    
    try:
        result = scrape_with_ai(
            "https://elitedatascience.com/learn-machine-learning",
            extraction_prompt="Extract information about the page, title, and content."
        )
        print_summary(result)
        save_results(result, "gemini_scraped_result.json")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*70 + "\n")
    
    # Example 2: AI-powered crawling
    print("Example 2: AI-Powered Website Crawling (Gemini)")
    print("-" * 70)
    
    try:
        results = crawl_with_ai(
            "https://elitedatascience.com/learn-machine-learning",
            max_pages=3,
            crawl_prompt="Find pages with unique information"
        )
        
        print(f"\nCrawled {len(results)} pages\n")
        
        for i, page in enumerate(results, 1):
            print(f"Page {i}: {page.get('url', 'N/A')}")
            if page.get('ai_summary'):
                print(f"  Summary: {page['ai_summary'][:150]}...")
            print()
        
        save_results(results, "gemini_crawled_results.json")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*70 + "\n")
    
    # Example 3: Structured data extraction
    print("Example 3: Structured Data Extraction (Gemini)")
    print("-" * 70)
    
    try:
        data = extract_data(
            "https://elitedatascience.com/learn-machine-learning",
            schema="books and techniques the page is describing.",
            api_key=api_key
        )
        
        if data.get('extracted_data'):
            print("Extracted Data:")
            print(json.dumps(data['extracted_data'], indent=2))
            save_results(data, "gemini_extracted_data.json")
        else:
            print(f"Error: {data.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"Error: {e}")


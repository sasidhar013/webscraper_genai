# AI Web Scraper using Google Gemini

AI-powered web scraper using Google Gemini AI instead of OpenAI. This version is designed for users who want to use Google's Gemini API.

## Features

- AI-powered content extraction and summarization
- Intelligent crawling decisions using Gemini AI
- Structured data extraction using AI
- Natural language queries

## Installation

1. Install required packages:
```bash
pip install -r requirements_gemini.txt
```

2. Get a Google API Key:
   - Go to https://makersuite.google.com/app/apikey
   - Create a new API key
   - Copy the key

3. Set up your API key (choose one method):

   **Option 1: .env file (Recommended)**
   - Create a `.env` file in the project root
   - Add: `GOOGLE_API_KEY=your-api-key-here`

   **Option 2: Environment Variable**
   - Windows PowerShell: `$env:GOOGLE_API_KEY='your-api-key-here'`
   - Windows CMD: `set GOOGLE_API_KEY=your-key-here`
   - Linux/Mac: `export GOOGLE_API_KEY='your-key-here'`

## Usage

### Basic Example

```python
from webscrapingusing_gemini import scrape_with_ai

result = scrape_with_ai(
    "https://example.com",
    extraction_prompt="Extract the main content and key information."
)

print(result['ai_summary'])
```

### Crawling Multiple Pages

```python
from webscrapingusing_gemini import crawl_with_ai

results = crawl_with_ai(
    "https://example.com",
    max_pages=5,
    crawl_prompt="Find pages with product information."
)
```

### Structured Data Extraction

```python
from webscrapingusing_gemini import extract_data

data = extract_data(
    "https://example.com",
    schema="Extract: title, price, description, and availability."
)
```

## Available Models

- `gemini-1.5-flash` (default, faster, good for most tasks)
- `gemini-1.5-pro` (more capable, better for complex tasks)
- `gemini-pro` (older model)

## Differences from OpenAI Version

- Uses `GOOGLE_API_KEY` instead of `OPENAI_API_KEY`
- Uses `google-generativeai` package instead of `openai`
- Model names are different (gemini-* instead of gpt-*)
- API response format is slightly different but functionality is the same

"# -webscraper_genai" 

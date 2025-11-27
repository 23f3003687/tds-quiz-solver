from dotenv import load_dotenv
load_dotenv()

import os
import json
import re
import base64
import requests
import asyncio
import subprocess
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from playwright.sync_api import sync_playwright
import google.generativeai as genai
import pandas as pd
from io import BytesIO, StringIO
import pdfplumber

# Install Playwright browsers if not present
try:
    subprocess.run(["playwright", "install", "chromium"], check=True, capture_output=True)
except:
    pass

# ---------------------------------------------------------------------------
# ENVIRONMENT VARIABLES
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
STUDENT_EMAIL = os.getenv("STUDENT_EMAIL", "23f3003687@ds.study.iitm.ac.in")
STUDENT_SECRET = os.getenv("STUDENT_SECRET", "moon")

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY environment variable")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# Thread pool for running sync Playwright
executor = ThreadPoolExecutor(max_workers=3)

# ---------------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------------
app = FastAPI()

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str
    model_config = ConfigDict(extra="ignore")

# ---------------------------------------------------------------------------
# BROWSER: FETCH RENDERED HTML (SYNC - runs in thread)
# ---------------------------------------------------------------------------
def fetch_html_sync(url: str) -> str:
    """Fetch fully rendered HTML using Playwright (sync version)."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=60000)
        page.wait_for_load_state("networkidle")
        html = page.content()
        browser.close()
        return html

async def fetch_html(url: str) -> str:
    """Run sync Playwright in thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, fetch_html_sync, url)

# ---------------------------------------------------------------------------
# EXTRACT QUIZ INFO FROM HTML
# ---------------------------------------------------------------------------
def extract_quiz_info(html: str, base_url: str) -> dict:
    """Use LLM to extract question, submit URL, and data sources from HTML."""
    from urllib.parse import urlparse
    base_domain = f"{urlparse(base_url).scheme}://{urlparse(base_url).netloc}"
    
    prompt = f"""Analyze this HTML and extract quiz information. Return ONLY valid JSON.

The base URL is: {base_url}
The base domain is: {base_domain}

IMPORTANT RULES:
1. For submit_url: Find where the answer should be POSTed. Convert relative URLs (like /submit) to full URLs using the base domain.
2. For data_sources: Find ALL URLs of files/pages to download/scrape - including CSV, audio files (.opus, .mp3, .wav), PDFs, etc. Convert relative URLs to full URLs.
3. IGNORE example.com URLs - those are just examples in the instructions, not real URLs.
4. Use the ACTUAL domain from the base_url ({base_domain}) for all URLs.
5. Look for audio elements, source tags, and href/src attributes for media files.
6. Look for any cutoff, threshold, or filter values mentioned in the HTML (in spans, divs, or text).

Return this JSON format:
{{
    "question": "the full question text including any cutoff/threshold values found",
    "submit_url": "full URL starting with {base_domain}",
    "data_sources": ["list of ALL file URLs including audio, csv, pdf, etc."],
    "answer_type": "number|string|boolean|json|base64",
    "cutoff_value": "any numeric cutoff/threshold value found in the HTML, or null if not found",
    "api_headers": {{"header_name": "header_value"}} or null if no special headers needed
}}

HTML:
""" + html[:15000]

    response = model.generate_content(prompt)
    text = response.text
    
    try:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            
            # Fix relative URLs
            from urllib.parse import urljoin
            if result.get("submit_url") and not result["submit_url"].startswith("http"):
                result["submit_url"] = urljoin(base_url, result["submit_url"])
            
            fixed_sources = []
            for src in result.get("data_sources", []):
                if not src.startswith("http"):
                    src = urljoin(base_url, src)
                fixed_sources.append(src)
            result["data_sources"] = fixed_sources
            
            return result
    except:
        pass
    
    raise RuntimeError(f"Failed to parse quiz info: {text[:500]}")

# ---------------------------------------------------------------------------
# DOWNLOAD FILE
# ---------------------------------------------------------------------------
def download_file(url: str) -> bytes:
    """Download file from URL."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content

# ---------------------------------------------------------------------------
# EXTRACT TEXT FROM PDF
# ---------------------------------------------------------------------------
def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text content from PDF."""
    text_parts = []
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text_parts.append(f"--- Page {i+1} ---")
            text_parts.append(page.extract_text() or "")
            
            tables = page.extract_tables()
            for j, table in enumerate(tables):
                text_parts.append(f"\nTable {j+1}:")
                for row in table:
                    text_parts.append(" | ".join(str(cell) if cell else "" for cell in row))
    
    return "\n".join(text_parts)

# ---------------------------------------------------------------------------
# EXTRACT CSV/EXCEL DATA
# ---------------------------------------------------------------------------
def extract_tabular_data(data: bytes, filename: str) -> str:
    """Extract data from CSV or Excel files."""
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(data))
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(BytesIO(data))
        else:
            try:
                df = pd.read_csv(BytesIO(data))
            except:
                df = pd.read_excel(BytesIO(data))
        
        # Include useful statistics
        result = f"Columns: {list(df.columns)}\n"
        result += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n"
        
        # For numeric columns, add statistics
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                result += f"Column '{col}' stats:\n"
                result += f"  Sum of ALL values: {df[col].sum()}\n"
                result += f"  Min: {df[col].min()}, Max: {df[col].max()}\n"
                
                # If column name is a number, calculate sum >= that number
                try:
                    cutoff = float(col)
                    values_above = df[df[col] >= cutoff][col]
                    result += f"  Sum of values >= {cutoff}: {values_above.sum()}\n"
                    result += f"  Count of values >= {cutoff}: {len(values_above)}\n"
                except:
                    pass
        
        result += f"\nFull Data:\n{df.to_string()}"
        return result
    except Exception as e:
        return f"Failed to parse tabular data: {e}"

# ---------------------------------------------------------------------------
# SOLVE QUIZ WITH CONTEXT
# ---------------------------------------------------------------------------
def solve_quiz(question: str, context: str = "") -> str:
    """Use LLM to solve the quiz question with given context."""
    prompt = f"""You are a quiz-solving AI. Your job is to answer the question using the provided context data.

QUESTION:
{question}

{"DATA/CONTEXT:" + chr(10) + context if context else ""}

CRITICAL RULES:
1. ONLY output the final answer - no explanations
2. For secret codes: extract from HTML and return just the code
3. For CSV data analysis:
   - If the column header is a NUMBER (like "96903"), that IS the cutoff value
   - Parse all the data values from the column
   - Apply the condition (>=, >, <, <=) using the cutoff
   - Sum/count the matching values
   - Return ONLY the numeric result
4. Do the actual calculation - don't just return the column header
5. Numbers: return plain numbers only (e.g., 42 or 3.14)
6. Strings: plain text without quotes
7. Boolean: true or false (lowercase)
8. If text is REVERSED (like "!dlroWolleH"), reverse it back (becomes "HelloWorld!")
9. For JavaScript code, EXECUTE the logic mentally:
   - [10, 20, 30].reduce((a,b) => a+b, 0) = 60
   - 60 * 2 = 120
10. If you see hidden-key or reversed text, reverse it to get the answer

EXAMPLE:
If column header is "96903" and values are [74775, 23534, 98000, 97000]
And condition is "sum values >= 96903"
Then: 98000 + 97000 = 195000
Answer: 195000

EXAMPLE 2:
If hidden text is "!dlroWolleH" and question asks for un-reversed password:
Reverse it: HelloWorld!
Answer: HelloWorld!

Now calculate the answer for the given question:

ANSWER (just the value, nothing else):"""

    response = model.generate_content(prompt)
    answer = response.text.strip()
    
    # Clean up common issues
    answer = answer.strip('`').strip('"').strip("'")
    answer = answer.replace("```", "").strip()
    if answer.lower() in ['true', 'false']:
        answer = answer.lower()
    
    # Remove common prefixes the LLM might add
    prefixes_to_remove = ["The answer is", "Answer:", "Result:", "The secret code is", "The sum is"]
    for prefix in prefixes_to_remove:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip().strip(':').strip()
    
    return answer

# ---------------------------------------------------------------------------
# PROCESS DATA SOURCE
# ---------------------------------------------------------------------------
def process_data_source_sync(url: str, headers: dict = None) -> str:
    """Download and process a data source URL (sync version for thread pool)."""
    try:
        filename = url.split('/')[-1].split('?')[0].lower()
        
        # For HTML pages that might need JS rendering, use Playwright
        if not filename.endswith(('.pdf', '.csv', '.xlsx', '.xls', '.json', '.txt', '.md', '.opus', '.mp3', '.wav')):
            # Use Playwright to render the page
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, timeout=60000)
                page.wait_for_load_state("networkidle")
                content = page.content()
                browser.close()
                return content
        
        # For API endpoints with pagination, handle specially
        if 'page=' in url or '/api/' in url:
            all_data = []
            current_url = url
            page_num = 1
            
            while current_url and page_num <= 100:  # Safety limit
                try:
                    resp = requests.get(current_url, headers=headers, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                    
                    if isinstance(data, list):
                        if len(data) == 0:
                            break  # Empty page, stop
                        all_data.extend(data)
                        # Try next page
                        if 'page=' in current_url:
                            page_num += 1
                            current_url = re.sub(r'page=\d+', f'page={page_num}', url)
                        else:
                            break
                    else:
                        return json.dumps(data, indent=2)
                except:
                    break
            
            if all_data:
                return json.dumps(all_data, indent=2)
        
        # For files, download directly
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.content
        
        if filename.endswith('.pdf'):
            return extract_pdf_text(data)
        elif filename.endswith(('.csv', '.xlsx', '.xls')):
            return extract_tabular_data(data, filename)
        elif filename.endswith('.json'):
            return json.dumps(json.loads(data), indent=2)
        elif filename.endswith(('.txt', '.md')):
            return data.decode('utf-8')
        elif filename.endswith(('.opus', '.mp3', '.wav', '.ogg', '.flac', '.m4a')):
            # Try to transcribe audio using Gemini
            try:
                import base64
                audio_b64 = base64.b64encode(data).decode('utf-8')
                
                # Determine mime type
                mime_types = {
                    '.opus': 'audio/opus',
                    '.mp3': 'audio/mp3',
                    '.wav': 'audio/wav',
                    '.ogg': 'audio/ogg',
                    '.flac': 'audio/flac',
                    '.m4a': 'audio/mp4'
                }
                ext = '.' + filename.split('.')[-1]
                mime_type = mime_types.get(ext, 'audio/mpeg')
                
                audio_prompt = """Transcribe this audio EXACTLY word for word. 
Include every number mentioned. Do not summarize - give the complete verbatim transcription."""
                
                audio_response = model.generate_content([
                    audio_prompt,
                    {"mime_type": mime_type, "data": audio_b64}
                ])
                return f"Audio transcription: {audio_response.text}"
            except Exception as e:
                return f"[Audio file - transcription failed: {e}]"
        else:
            try:
                return data.decode('utf-8')
            except:
                return f"Binary file: {len(data)} bytes"
    except Exception as e:
        return f"Error processing {url}: {e}"

async def process_data_source(url: str, headers: dict = None) -> str:
    """Run data source processing in thread pool."""
    loop = asyncio.get_event_loop()
    import functools
    func = functools.partial(process_data_source_sync, url, headers)
    return await loop.run_in_executor(executor, func)

# ---------------------------------------------------------------------------
# PARSE ANSWER TO CORRECT TYPE
# ---------------------------------------------------------------------------
def parse_answer(answer: str, answer_type: str):
    """Convert answer string to the appropriate type."""
    answer = answer.strip()
    
    if answer_type == "number":
        num_match = re.search(r'-?\d+\.?\d*', answer)
        if num_match:
            num_str = num_match.group()
            return float(num_str) if '.' in num_str else int(num_str)
        return answer
    
    elif answer_type == "boolean":
        return answer.lower() == "true"
    
    elif answer_type == "json":
        try:
            return json.loads(answer)
        except:
            return answer
    
    return answer

# ---------------------------------------------------------------------------
# SUBMIT ANSWER
# ---------------------------------------------------------------------------
def submit_answer(submit_url: str, original_url: str, answer):
    """Submit answer to the quiz endpoint."""
    payload = {
        "email": STUDENT_EMAIL,
        "secret": STUDENT_SECRET,
        "url": original_url,
        "answer": answer
    }
    
    resp = requests.post(submit_url, json=payload, timeout=30)
    try:
        return resp.json()
    except:
        return {"correct": False, "reason": f"Invalid response: {resp.text[:200]}"}

# ---------------------------------------------------------------------------
# MAIN QUIZ SOLVER (ASYNC)
# ---------------------------------------------------------------------------
async def solve_quiz_url(url: str) -> dict:
    """Complete flow: fetch, parse, solve, submit."""
    print(f"\n{'='*50}")
    print(f"[DEBUG] Fetching URL: {url}")
    
    html = await fetch_html(url)
    print(f"[DEBUG] HTML length: {len(html)} chars")
    print(f"[DEBUG] Full HTML:\n{html[:2000]}")  # Show more HTML
    
    quiz_info = extract_quiz_info(html, url)
    question = quiz_info.get("question", "")
    submit_url = quiz_info.get("submit_url", "")
    data_sources = quiz_info.get("data_sources", [])
    answer_type = quiz_info.get("answer_type", "string")
    
    print(f"[DEBUG] Question: {question[:200]}")
    print(f"[DEBUG] Submit URL: {submit_url}")
    print(f"[DEBUG] Data sources: {data_sources}")
    print(f"[DEBUG] Answer type: {answer_type}")
    
    # Get cutoff value if present
    cutoff_value = quiz_info.get("cutoff_value")
    print(f"[DEBUG] Cutoff value: {cutoff_value}")
    
    # Get API headers if present
    api_headers = quiz_info.get("api_headers") or {}
    if api_headers:
        print(f"[DEBUG] API headers: {api_headers}")
    
    if not submit_url:
        raise RuntimeError("Could not find submit URL")
    
    context = ""
    for source_url in data_sources:
        context += f"\n\n=== Data from {source_url} ===\n"
        source_content = await process_data_source(source_url, api_headers)
        
        # If we have a cutoff value and this is CSV data, add filtered calculations
        if cutoff_value and '.csv' in source_url.lower():
            try:
                cutoff_num = float(cutoff_value)
                context += f"\n[IMPORTANT: The cutoff value from the page is {cutoff_num}]\n"
                # Recalculate with the correct cutoff
                resp = requests.get(source_url, timeout=30)
                
                # Try reading with header
                df_with_header = pd.read_csv(BytesIO(resp.content))
                # Try reading without header
                df_no_header = pd.read_csv(BytesIO(resp.content), header=None)
                
                context += f"\n--- With header (first row as column name) ---\n"
                for col in df_with_header.columns:
                    if pd.api.types.is_numeric_dtype(df_with_header[col]):
                        values_above = df_with_header[df_with_header[col] >= cutoff_num][col]
                        context += f"Sum of values >= {cutoff_num} (with header): {values_above.sum()}\n"
                
                context += f"\n--- Without header (first row as data) - USE THIS ---\n"
                for col in df_no_header.columns:
                    if pd.api.types.is_numeric_dtype(df_no_header[col]):
                        values_above = df_no_header[df_no_header[col] >= cutoff_num][col]
                        context += f"[CORRECT ANSWER] Sum of values >= {cutoff_num}: {values_above.sum()}\n"
                        context += f"All values sum: {df_no_header[col].sum()}\n"
            except Exception as e:
                context += f"Error calculating: {e}\n"
        
        context += source_content
    
    if context:
        print(f"[DEBUG] Context length: {len(context)} chars")
        print(f"[DEBUG] Context preview: {context[:500]}")
    
    raw_answer = solve_quiz(question, context)
    answer = parse_answer(raw_answer, answer_type)
    
    print(f"[DEBUG] Raw answer: {raw_answer}")
    print(f"[DEBUG] Parsed answer: {answer}")
    
    result = submit_answer(submit_url, url, answer)
    print(f"[DEBUG] Result: {result}")
    print(f"{'='*50}\n")
    
    return {
        "question": question,
        "answer": answer,
        "result": result
    }

# ---------------------------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------------------------
@app.post("/")
async def handle_quiz(request: Request):
    """Main endpoint to receive and process quiz tasks."""
    try:
        body = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    task = QuizRequest(**body)
    
    if task.secret != STUDENT_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    try:
        result = await solve_quiz_url(task.url)
        
        final_result = result["result"]
        while final_result.get("url"):
            next_url = final_result["url"]
            next_result = await solve_quiz_url(next_url)
            final_result = next_result["result"]
        
        return final_result
        
    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={"error": str(e), "status": "failed"}
        )

@app.get("/")
def home():
    """Health check endpoint."""
    return {"status": "running", "email": STUDENT_EMAIL}

@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}
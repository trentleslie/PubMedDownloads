import os
import sqlite3
import requests
import time
import csv
from Bio import Entrez
import re
from docx import Document
from fuzzywuzzy import fuzz, process
import PyPDF2
import shutil
import json
import logging
import hashlib
from tqdm import tqdm
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
from api_key import openai_api_key

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables and set up OpenAI client
load_dotenv()
client = OpenAI(api_key=openai_api_key)

# Set email for Entrez
Entrez.email = "trentleslie@gmail.com"  # Replace with your email

def extract_references_from_docx(file_path):
    doc = Document(file_path)
    full_text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    # Find the "References" section
    references_start = full_text.find("References")
    if references_start == -1:
        logger.warning("References section not found in the document.")
        return []
    
    references_text = full_text[references_start:]
    
    # Split the references into individual entries
    raw_references = re.split(r'\n\d+\.', references_text)[1:]  # Skip the "References" header
    
    cleaned_references = [ref.strip() for ref in raw_references if ref.strip()]
    
    logger.info(f"Extracted {len(cleaned_references)} references from the document.")
    return cleaned_references

def get_cache_key(reference):
    return hashlib.md5(reference.encode()).hexdigest()

def clean_extracted_info(info):
    # Clean up authors
    if isinstance(info['authors'], list):
        info['authors'] = ', '.join(author.strip() for author in info['authors'] if author.strip())
    elif isinstance(info['authors'], str):
        info['authors'] = info['authors'].replace('[', '').replace(']', '').replace("'", "")
    
    # Clean up year
    if info['year']:
        info['year'] = info['year'].strip()
    
    # Clean up DOI
    if info['doi']:
        info['doi'] = info['doi'].strip().lower()
        if info['doi'].startswith('doi:'):
            info['doi'] = info['doi'][4:]
    
    # Clean up journal info
    if isinstance(info['journal_info'], dict):
        journal_parts = []
        if info['journal_info'].get('journal_name'):
            journal_parts.append(info['journal_info']['journal_name'])
        if info['journal_info'].get('volume'):
            journal_parts.append(info['journal_info']['volume'])
        if info['journal_info'].get('issue'):
            journal_parts.append(info['journal_info']['issue'])
        if info['journal_info'].get('pages'):
            journal_parts.append(info['journal_info']['pages'])
        info['journal_info'] = ', '.join(part for part in journal_parts if part)
    elif isinstance(info['journal_info'], str):
        info['journal_info'] = info['journal_info'].replace('{', '').replace('}', '')
    
    # Remove any "et al." from the title
    info['title'] = re.sub(r'\s+et al\.?$', '', info['title'])
    
    return info

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def extract_reference_info(reference, cache={}):
    cache_key = get_cache_key(reference)
    if cache_key in cache:
        logger.info("Using cached result for reference")
        return cache[cache_key]
    
    system_message = """
    You are an expert in parsing academic references. Your task is to extract key information from a given reference.
    Please extract the following fields:
    - title: The title of the paper or book
    - authors: List of authors (last name, first initial)
    - year: Year of publication
    - doi: Digital Object Identifier (if available)
    - journal_info: Journal name, volume, issue, pages

    Format the output as a JSON object with these keys.
    For authors, provide a simple comma-separated string, not a list.
    If a field is not present in the reference, use an empty string for its value.
    """

    user_message = f"Parse this reference and extract the required information: {reference}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
        )
        extracted_info = json.loads(response.choices[0].message.content)
        cleaned_info = clean_extracted_info(extracted_info)
        cache[cache_key] = cleaned_info
        return cleaned_info
    except Exception as e:
        logger.error(f"API request failed: {e}")
        raise

def clean_extracted_info(info):
    # Remove any leading/trailing whitespace
    for key in info:
        if isinstance(info[key], str):
            info[key] = info[key].strip()
    
    # Remove any "et al." from the title
    info['title'] = re.sub(r'\s+et al\.?$', '', info['title'])
    
    # Ensure the year is a 4-digit number
    if info['year'] and not re.match(r'^\d{4}$', info['year']):
        info['year'] = ''
    
    return info

def create_pubmed_strategies(ref_info):
    title_words = ref_info["title"].split()
    short_title = " ".join(title_words[:8])  # Use first 8 words of title
    
    authors = ref_info["authors"].split(',')
    first_author = authors[0].split()[-1] if authors else ""
    
    return [
        lambda: f'"{short_title}"[Title] AND {first_author}[Author] AND {ref_info["year"]}[Date - Publication]',
        lambda: f'"{short_title}"[Title] AND {ref_info["year"]}[Date - Publication]',
        lambda: f'{first_author}[Author] AND {ref_info["year"]}[Date - Publication]',
        lambda: f'"{" ".join(title_words[:5])}"[Title] AND {ref_info["year"]}[Date - Publication]',
        lambda: f'"{short_title}"[Title]',
        lambda: f'{first_author}[Author] AND "{" ".join(title_words[:5])}"[Title]',
        lambda: f'{ref_info.get("doi", "")}[AID]' if ref_info.get("doi") else None,
        lambda: f'"{ref_info.get("journal_info", "")}"[Journal] AND {ref_info["year"]}[Date - Publication] AND {first_author}[Author]' if ref_info.get("journal_info") else None,
        lambda: ' AND '.join([f'"{word}"[Title/Abstract]' for word in title_words[:5] if len(word) > 3]) + f' AND {ref_info["year"]}[Date - Publication]'
    ]

def search_pubmed(query, max_results=20, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            handle = Entrez.esearch(db='pubmed', 
                                    sort='relevance', 
                                    retmax=max_results,
                                    retmode='xml', 
                                    term=query)
            results = Entrez.read(handle)
            handle.close()
            return results
        except Exception as e:
            print(f"Error in search_pubmed: {e}")
            time.sleep(1)  # Wait for 1 second before retrying
    print(f"PubMed search timed out after {timeout} seconds")
    return None

def get_pmc_id(article):
    try:
        for id_obj in article['PubmedData']['ArticleIdList']:
            if id_obj.attributes['IdType'] == 'pmc':
                return str(id_obj)
    except KeyError:
        pass
    return None

def sanitize_filename(filename):
    # Remove invalid characters
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, '', filename)
    
    # Truncate to a maximum length (e.g., 200 characters)
    max_length = 200
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length-4] + '...'
    
    return sanitized

def clean_text(text):
    return re.sub(r'[^\w\s]', '', text.lower())

def search_local_database(conn, ref_info):
    c = conn.cursor()
    
    conditions = []
    params = []
    
    # Title search
    if ref_info['title']:
        title_words = ref_info['title'].lower().split()
        title_condition = ' AND '.join(['LOWER(title) LIKE ?' for _ in title_words])
        conditions.append(f"({title_condition})")
        params.extend([f'%{word}%' for word in title_words])
    
    # Author search
    if ref_info['authors']:
        author_last_names = [name.split()[-1].lower() for name in ref_info['authors'].split(',') if name.strip()]
        author_condition = ' OR '.join(['LOWER(authors) LIKE ?' for _ in author_last_names])
        conditions.append(f"({author_condition})")
        params.extend([f'%{name}%' for name in author_last_names])
    
    # Year search
    if ref_info['year']:
        conditions.append("year = ?")
        params.append(ref_info['year'])
    
    # DOI search
    if ref_info['doi']:
        conditions.append("doi LIKE ?")
        params.append(f"%{ref_info['doi']}%")

    if not conditions:
        print("No valid search criteria. Skipping database search.")
        return []

    query = f"SELECT * FROM articles WHERE {' AND '.join(conditions)} LIMIT 50"

    print(f"\nSearching local database with query: {query}")
    print(f"Parameters: {params}")

    try:
        c.execute(query, params)
        results = c.fetchall()
        print(f"Found {len(results)} potential matches in the local database.")
        
        # Compare and sort results
        scored_results = []
        for result in results:
            score = compare_reference(ref_info, result)
            scored_results.append((result, score))
            print(f"PMID: {result[0]}, Title: {result[2]}, Score: {score:.2f}")
        
        sorted_results = sorted(scored_results, key=lambda x: x[1], reverse=True)
        
        # Filter results with a minimum score (e.g., 60%)
        filtered_results = [result for result, score in sorted_results if score >= 60]
        
        print(f"After filtering, {len(filtered_results)} relevant matches remain.")
        
        return filtered_results
    except sqlite3.Error as e:
        print(f"An error occurred while searching the database: {e}")
        return []

def search_pubmed(query, max_results=20):
    try:
        handle = Entrez.esearch(db='pubmed', 
                                sort='relevance', 
                                retmax=max_results,
                                retmode='xml', 
                                term=query)
        results = Entrez.read(handle)
        handle.close()

        # If no results and query contains group author, try without parentheses
        if int(results['Count']) == 0 and '(' in query and ')' in query:
            query_without_parentheses = re.sub(r'\([^)]*\)', '', query)
            handle = Entrez.esearch(db='pubmed', 
                                    sort='relevance', 
                                    retmax=max_results,
                                    retmode='xml', 
                                    term=query_without_parentheses)
            results = Entrez.read(handle)
            handle.close()

        return results
    except Exception as e:
        print(f"Error in search_pubmed: {e}")
        return None

def fetch_article_details(pubmed_id):
    try:
        handle = Entrez.efetch(db='pubmed', id=pubmed_id, rettype='xml', retmode='xml')
        records = Entrez.read(handle)
        handle.close()
        if records['PubmedArticle']:
            article = records['PubmedArticle'][0]
            journal_name = article['MedlineCitation']['Article']['Journal']['Title']
            return article, journal_name
        else:
            print(f"No article details found for PubMed ID: {pubmed_id}")
            return None, None
    except Exception as e:
        print(f"Error in fetch_article_details for PubMed ID {pubmed_id}: {e}")
        return None, None

def download_pdf(pmc_id, output_folder):
    url = f'https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            if 'application/pdf' in content_type:
                pdf_filename = os.path.join(output_folder, f'{pmc_id}.pdf')
                with open(pdf_filename, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded: {pdf_filename}")
                return pdf_filename, "Success"
            else:
                return None, "Not a PDF (possibly HTML article)"
        elif response.status_code == 403:
            return None, "Access denied (possibly paywalled)"
        elif response.status_code == 404:
            return None, "PDF not found"
        else:
            return None, f"Failed to download (Status Code: {response.status_code})"
    except Exception as e:
        return None, f"Error downloading PDF: {str(e)}"
        return None

def extract_title_from_pdf(pdf_text):
    system_message = """
    You are an expert in extracting article titles from academic PDFs. Your task is to identify and extract the main title of the academic article from the given text, which is typically from the first few pages of a PDF.
    Please extract the following:
    - title: The main title of the academic article

    Format the output as a JSON object with this key.
    If you cannot find a title, use an empty string for its value.
    """

    user_message = f"Extract the main title from this text: {pdf_text[:1000]}"  # Use first 1000 characters

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
        )
        extracted_info = json.loads(response.choices[0].message.content)
        return extracted_info.get('title', '')
    except Exception as e:
        print(f"Error extracting title from PDF: {e}")
        return ''

def compare_titles(title1, title2):
    system_message = """
    You are an expert in comparing academic article titles. Your task is to determine if two given titles refer to the same academic article, accounting for minor variations in formatting or wording.
    Please provide:
    - match: A boolean indicating whether the titles match (true) or not (false)
    - confidence: A float between 0 and 1 indicating your confidence in the match decision

    Format the output as a JSON object with these keys.
    """

    user_message = f"Compare these two titles and determine if they refer to the same article:\nTitle 1: {title1}\nTitle 2: {title2}"

    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return result.get('match', False), result.get('confidence', 0)
    except Exception as e:
        print(f"Error comparing titles: {e}")
        return False, 0

def verify_pdf(pdf_path, ref_info):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages[:5]:  # Check first 5 pages
                text += page.extract_text()
        
        pdf_title = extract_title_from_pdf(text)
        title_match, confidence = compare_titles(pdf_title, ref_info['title'])

        print(f"PDF Title: {pdf_title}")
        print(f"Reference Title: {ref_info['title']}")
        print(f"Title match: {title_match} (confidence: {confidence:.2f})")

        if title_match and confidence > 0.8:
            return True
        elif title_match and confidence > 0.6:
            text = text.lower()
            year_match = ref_info['year'] in text
            author_last_names = [name.split()[-1].lower() for name in ref_info['authors'].split(',') if name.strip()]
            author_match = any(name in text for name in author_last_names)
            journal_match = ref_info['journal_info'].lower() in text

            print(f"Year match: {year_match}")
            print(f"Author match: {author_match}")
            print(f"Journal match: {journal_match}")

            if year_match and (author_match or journal_match):
                return True
            else:
                return "Moderate title match, but year/author/journal mismatch"
        else:
            return "Title mismatch"

    except Exception as e:
        return f"Error verifying PDF: {str(e)}"

def clean_reference(reference):
    # Remove any leading/trailing whitespace
    reference = reference.strip()
    # Replace multiple spaces with a single space
    reference = re.sub(r'\s+', ' ', reference)
    # Ensure there's a space after each comma
    reference = re.sub(r',(?!\s)', ', ', reference)
    return reference

def search_and_download(conn, ref_info, reference, output_folder, rejected_folder, downloaded_pmcs):
    print(f"\nProcessing reference: {reference}")
    print(f"Extracted info: {json.dumps(ref_info, indent=2)}")

    pubmed_strategies = create_pubmed_strategies(ref_info)
    journal_name = ""  # Initialize journal name

    for strategy in pubmed_strategies:
        query = strategy()
        if query:
            print(f"Trying PubMed search strategy: {query}")
            results = search_pubmed(query)
            if results is None:
                continue
            if results and int(results['Count']) > 0:
                print(f"Found {results['Count']} results")
                for pubmed_id in results['IdList']:
                    article, journal = fetch_article_details(pubmed_id)
                    if article:
                        journal_name = journal  # Store the journal name
                        pmc_id = get_pmc_id(article)
                        if pmc_id:
                            pdf_filename, status = download_and_verify_pdf(pmc_id, ref_info, output_folder, rejected_folder, downloaded_pmcs)
                            if pdf_filename:
                                return pdf_filename, status, journal_name
                            else:
                                print(f"  {status}")
                        else:
                            print("  No PMC ID available for this article")
                    else:
                        print("  Failed to fetch article details")
            else:
                print("No results found for this strategy")
        time.sleep(1)  # Add a small delay between requests to avoid overwhelming the API

    return None, "All strategies exhausted. No matching PDF found.", journal_name

def download_and_verify_pdf(pmc_id, ref_info, output_folder, rejected_folder, downloaded_pmcs):
    print(f"  PMC ID: {pmc_id}")
    if pmc_id in downloaded_pmcs:
        return None, "PDF already downloaded for this PMC ID"
    
    pdf_filename, download_status = download_pdf(pmc_id, output_folder)
    if pdf_filename:
        print(f"  Downloaded PDF: {pdf_filename}")
        downloaded_pmcs.add(pmc_id)
        verification_result = verify_pdf(pdf_filename, ref_info)
        if verification_result == True:
            new_filename = os.path.join(output_folder, f"{sanitize_filename(ref_info['title'])}.pdf")
            os.rename(pdf_filename, new_filename)
            print(f"  Verified and renamed: {pdf_filename} -> {new_filename}")
            return new_filename, "PDF downloaded, verified, and renamed"
        else:
            rejected_filename = os.path.join(rejected_folder, f"{sanitize_filename(ref_info['title'])}_{pmc_id}.pdf")
            shutil.move(pdf_filename, rejected_filename)
            print(f"  Verification failed. Moved to: {rejected_filename}")
            return None, f"Verification failed: {verification_result}"
    else:
        return None, download_status

def compare_reference(ref_info, db_result):
    score = 0
    max_score = 0

    if ref_info['title'] and db_result[2]:
        title_similarity = fuzz.token_set_ratio(ref_info['title'].lower(), db_result[2].lower())
        score += title_similarity * 2  # Give more weight to title matches
        max_score += 200

    if ref_info['authors'] and db_result[1]:
        authors_similarity = fuzz.token_set_ratio(ref_info['authors'].lower(), db_result[1].lower())
        score += authors_similarity
        max_score += 100

    if ref_info['year'] and db_result[3]:
        year_similarity = 100 if ref_info['year'] == str(db_result[3]) else 0
        score += year_similarity
        max_score += 100

    return (score / max_score * 100) if max_score > 0 else 0

def process_references(docx_file_path, output_folder, rejected_folder, csv_output_file):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(rejected_folder, exist_ok=True)
    
    try:
        conn = sqlite3.connect('pubmed_local.db')
        logger.info("Connected to local PubMed database.")
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database: {e}")
        return
    
    references = extract_references_from_docx(docx_file_path)
    
    results = []
    downloaded_pmcs = set()  # Set to keep track of downloaded PMC IDs
    cache = {}  # Cache for extracted reference info
    
    for ref in tqdm(references, desc="Processing references"):
        cleaned_ref = clean_reference(ref)
        ref_info = extract_reference_info(cleaned_ref, cache)
        
        pdf_filename, status, journal_name = search_and_download(conn, ref_info, cleaned_ref, output_folder, rejected_folder, downloaded_pmcs)
        
        if pdf_filename:
            results.append({
                'Reference': ref, 
                'Downloaded': 'Yes', 
                'Reason': status, 
                'Filename': os.path.basename(pdf_filename),
                'Journal': journal_name
            })
        else:
            results.append({
                'Reference': ref, 
                'Downloaded': 'No', 
                'Reason': status, 
                'Filename': '',
                'Journal': journal_name
            })
    
    conn.close()
    logger.info("Closed connection to local PubMed database.")
    
    # Write results to CSV
    with open(csv_output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Reference', 'Downloaded', 'Reason', 'Filename', 'Journal']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    logger.info(f"Results saved to {csv_output_file}")

if __name__ == "__main__":
    docx_file_path = '57ISB15US - REFERENCES (1).docx'  # Replace with the path to your DOCX file
    output_folder = './pdf_downloads'  # Folder where verified PDFs will be saved
    rejected_folder = './rejected_pdfs'  # Folder for rejected PDFs
    csv_output_file = 'download_results.csv'  # CSV file to store results
    
    process_references(docx_file_path, output_folder, rejected_folder, csv_output_file)
import os
import csv
import json
import time
import hashlib
import logging
from tqdm import tqdm
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
from docx import Document
import re
from api_key import openai_api_key

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables and set up OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv(openai_api_key))

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
            model="gpt-4-1106-preview",
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

def validate_extracted_info(info):
    if not info['title'] or len(info['title']) < 5:
        logger.warning(f"Title seems too short or missing: {info['title']}")
    if not info['year'] or not info['year'].isdigit() or len(info['year']) != 4:
        logger.warning(f"Invalid year: {info['year']}")
    # Add more validation as needed

def process_references(docx_file_path, output_csv, batch_size=10):
    references = extract_references_from_docx(docx_file_path)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        fieldnames = ['Reference', 'Title', 'Authors', 'Year', 'DOI', 'Journal Info']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        cache = {}

        for i in tqdm(range(0, len(references), batch_size), desc="Processing batches"):
            batch = references[i:i+batch_size]
            for reference in batch:
                logger.info(f"Processing reference: {reference[:50]}...")  # Log first 50 chars
                extracted_info = extract_reference_info(reference, cache)
                
                if extracted_info:
                    validate_extracted_info(extracted_info)
                    writer.writerow({
                        'Reference': reference,
                        'Title': extracted_info['title'],
                        'Authors': extracted_info['authors'],
                        'Year': extracted_info['year'],
                        'DOI': extracted_info['doi'],
                        'Journal Info': extracted_info['journal_info']
                    })
                else:
                    logger.error(f"Failed to extract info for reference: {reference[:50]}...")

            # Add a small delay between batches to avoid rate limiting
            time.sleep(1)

if __name__ == "__main__":
    docx_file_path = '57ISB15US - REFERENCES (1).docx'  # Path to your DOCX file
    output_csv = 'processed_references.csv'  # Output CSV file path
    
    process_references(docx_file_path, output_csv)
    logger.info(f"Processed references saved to {output_csv}")
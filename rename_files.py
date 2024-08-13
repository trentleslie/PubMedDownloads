import os
import re
import csv
from openai import OpenAI
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from api_key import openai_api_key

# Load environment variables and set up OpenAI client
load_dotenv()
client = OpenAI(api_key=openai_api_key)

# Set up paths
pdf_directory = '/home/trent/github/PubMedDownloads/pdf_downloads_saved'
references_file = '/home/trent/github/PubMedDownloads/references.txt'
output_csv = 'file_renaming_summary.csv'

def extract_reference_info(reference):
    # Extract the sequence number and title from the reference
    match = re.match(r'^(\d{2})\.\s(.+)$', reference.strip())
    if match:
        return match.group(1), match.group(2)
    return None, None

def extract_title_from_filename(filename):
    # Remove file extension and any additional identifiers
    return os.path.splitext(filename)[0].split('_')[0]

def extract_title_using_openai(text):
    system_message = """
    You are an expert in extracting article titles. Your task is to identify and extract the main title of the academic article from the given text.
    Please extract the following:
    - title: The main title of the academic article

    Format the output as a JSON object with this key.
    If you cannot find a title, use an empty string for its value.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Extract the main title from this text: {text}"}
            ],
            response_format={"type": "json_object"},
        )
        extracted_info = response.choices[0].message.content
        return extracted_info.get('title', '')
    except Exception as e:
        print(f"Error extracting title using OpenAI: {e}")
        return ''

def rename_files():
    # Read references
    with open(references_file, 'r') as f:
        references = f.readlines()

    # Process each file in the directory
    results = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            old_filepath = os.path.join(pdf_directory, filename)
            file_title = extract_title_from_filename(filename)

            # Find matching reference
            matched_reference = None
            highest_similarity = 0
            for reference in references:
                seq_num, ref_title = extract_reference_info(reference)
                if seq_num and ref_title:
                    similarity = fuzz.ratio(file_title.lower(), ref_title.lower())
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        matched_reference = (seq_num, ref_title, reference.strip())

            if matched_reference:
                seq_num, ref_title, full_reference = matched_reference
                new_filename = f"{seq_num}_{file_title}.pdf"
                new_filepath = os.path.join(pdf_directory, new_filename)

                # Rename the file
                os.rename(old_filepath, new_filepath)

                results.append({
                    'Reference': full_reference,
                    'Old Filename': filename,
                    'New Filename': new_filename
                })
            else:
                # If no match found, use OpenAI to extract title
                openai_title = extract_title_using_openai(file_title)
                new_filename = f"unmatched_{openai_title}.pdf"
                new_filepath = os.path.join(pdf_directory, new_filename)
                
                # Rename the file
                os.rename(old_filepath, new_filepath)

                results.append({
                    'Reference': 'No match found',
                    'Old Filename': filename,
                    'New Filename': new_filename
                })

    # Write results to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Reference', 'Old Filename', 'New Filename']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"File renaming complete. Summary saved to {output_csv}")

# Run the renaming process
if __name__ == "__main__":
    rename_files()
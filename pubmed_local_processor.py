import os
import gzip
import xml.etree.ElementTree as ET
import sqlite3
from ftplib import FTP
import datetime
import time
from tqdm import tqdm

def download_pubmed_files(ftp_dir, local_dir, max_retries=3, retry_delay=5):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    for attempt in range(max_retries):
        try:
            with FTP('ftp.ncbi.nlm.nih.gov') as ftp:
                ftp.login()
                ftp.cwd(ftp_dir)
                ftp.voidcmd('TYPE I')  # Set binary mode
                
                files = ftp.nlst()
                for file in tqdm(files, desc="Processing files", unit="file"):
                    if file.endswith('.xml.gz'):
                        local_file = os.path.join(local_dir, file)
                        if not os.path.exists(local_file):
                            with open(local_file, 'wb') as f:
                                def callback(data):
                                    f.write(data)
                                    tqdm.write(f"\rDownloading {file}", end="")
                                
                                ftp.retrbinary(f'RETR {file}', callback)
                            
                            print(f"\nDownloaded {file}")
                        else:
                            print(f"Skipping {file} (already exists)")
            
            return  # Success, exit the function
        except Exception as e:
            print(f"Error during download: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Download failed.")
                raise

def create_database():
    conn = sqlite3.connect('pubmed_local.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS articles
                 (pmid TEXT PRIMARY KEY, title TEXT, authors TEXT, year INTEGER, journal TEXT, doi TEXT)''')
    conn.commit()
    return conn

def process_xml_file(file_path, conn):
    c = conn.cursor()
    try:
        with gzip.open(file_path, 'rb') as f:
            tree = ET.parse(f)
            root = tree.getroot()
            for article in root.findall('.//PubmedArticle'):
                pmid = article.find('.//PMID').text
                title = article.find('.//ArticleTitle').text
                authors = ', '.join([author.find('LastName').text for author in article.findall('.//Author') if author.find('LastName') is not None])
                year = article.find('.//PubDate/Year')
                year = int(year.text) if year is not None else None
                journal = article.find('.//Journal/Title').text
                doi = article.find('.//ELocationID[@EIdType="doi"]')
                doi = doi.text if doi is not None else None
                
                c.execute("INSERT OR REPLACE INTO articles VALUES (?, ?, ?, ?, ?, ?)",
                          (pmid, title, authors, year, journal, doi))
        
        conn.commit()
        print(f"Processed {file_path}")
    except ET.ParseError as e:
        print(f"Error parsing {file_path}: {e}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def update_database():
    today = datetime.date.today()
    current_year = today.year
    baseline_dir = f'/pubmed/baseline'
    update_dir = f'/pubmed/updatefiles'
    
    download_pubmed_files(baseline_dir, './pubmed_xml')
    download_pubmed_files(update_dir, './pubmed_xml_updates')
    
    conn = create_database()
    
    xml_dirs = ['./pubmed_xml', './pubmed_xml_updates']
    for xml_dir in xml_dirs:
        for file in os.listdir(xml_dir):
            if file.endswith('.xml.gz'):
                process_xml_file(os.path.join(xml_dir, file), conn)
    
    conn.close()

def search_local_database(query):
    conn = sqlite3.connect('pubmed_local.db')
    c = conn.cursor()
    
    # Implement various search strategies here
    c.execute("SELECT * FROM articles WHERE title LIKE ? OR authors LIKE ?", (f"%{query}%", f"%{query}%"))
    results = c.fetchall()
    
    conn.close()
    return results

if __name__ == "__main__":
    print("PubMed Local Processor")
    print("1. Update Database")
    print("2. Search Database")
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        print("Updating database... This may take several hours.")
        update_database()
        print("Database update completed.")
    elif choice == "2":
        query = input("Enter your search query: ")
        results = search_local_database(query)
        print(f"Found {len(results)} results:")
        for result in results[:10]:  # Display first 10 results
            print(f"PMID: {result[0]}, Title: {result[1]}, Authors: {result[2]}, Year: {result[3]}")
    else:
        print("Invalid choice. Please run the script again and choose 1 or 2.")
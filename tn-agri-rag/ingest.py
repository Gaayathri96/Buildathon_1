import os
import requests
from bs4 import BeautifulSoup
import pymupdf  # PyMuPDF for reading PDFs

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def read_pdfs_and_scrape_urls(pdf_folder_path):
    print("🚀 Step 1/2: Starting PDF reading and URL scraping...")
    all_documents = []
    
    if not os.path.exists(pdf_folder_path):
        print(f"❌ Error: Folder '{pdf_folder_path}' does not exist.")
        return all_documents
        
    pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"⚠️ Warning: No PDF files found in '{pdf_folder_path}'. Please add at least one PDF.")
        return all_documents
    
    for path in pdf_files:
        print(f"📄 Reading PDF: {path}")
        doc = pymupdf.open(path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Extract hyperlinks present on the page
            links = [link["uri"] for link in page.get_links() if "uri" in link]
            
            scraped_web_content = ""
            for url in links:
                if url.startswith("http"):
                    try:
                        print(f"🌐 Scraping URL found on page {page_num+1}: {url}")
                        response = requests.get(url, timeout=5)
                        if response.status_code == 200:
                            soup = BeautifulSoup(response.text, 'html.parser')
                            for script in soup(["script", "style"]):
                                script.extract()
                            scraped_web_content += f"\n\n[Web Reference from {url}]:\n{soup.get_text(separator=' ').strip()}"
                    except Exception as e:
                        print(f"⚠️ Could not scrape {url}: {e}")
            
            combined_content = text + scraped_web_content
            
            all_documents.append({
                "content": combined_content,
                "metadata": {"source": path, "page": page_num + 1}
            })
            
    print(f"✅ Extracted {len(all_documents)} pages from PDFs!")
    return all_documents


def create_vector_store(documents, persist_directory="./chroma_db"):
    print("\n🚀 Step 2/2: Chunking text and creating Vector Store...")
    
    # 1. Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    texts = []
    metadatas = []
    
    # 2. Split documents into chunks
    for doc in documents:
        chunks = text_splitter.split_text(doc["content"])
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append(doc["metadata"])
            
    print(f"✂️ Created {len(texts)} chunks from all documents.")
    
    # 3. Create OpenAI Embeddings & Store in ChromaDB
    print("🧠 Generating embeddings and saving to ChromaDB (this may take a few seconds)...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_directory
    )
    
    print(f"🎉 SUCCESS! Vector database saved locally at: '{persist_directory}'")
    return vectorstore


if __name__ == "__main__":
    extracted_docs = read_pdfs_and_scrape_urls("./tn_agri_pdfs")
    if extracted_docs:
        create_vector_store(extracted_docs)
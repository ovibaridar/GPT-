import os
import textract
from transformers import GPT2TokenizerFast
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

import Api_key

os.environ["OPENAI_API_KEY"] = Api_key.API_KEY

# Debugging: Print current working directory

# Construct absolute path
file_path = r'I:/elestator/GPT-/AI_Chatbot_Development/Data_update/merged_file.pdf'
# absolute_path = os.path.abspath(file_path)

loader = PyPDFLoader(file_path)

# Load and split pages
pages = loader.load_and_split()

# SKIP TO STEP 2 IF YOU'RE USING THIS METHOD
chunks = pages

# Advanced method - Split by chunk

# Step 1: Convert PDF to text

doc = textract.process(file_path)

# Step 2: Save to .txt and reopen (helps prevent issues)
with open('../../attention_is_all_you_need.txt', 'w', encoding='utf-8') as f:
    f.write(doc.decode('utf-8'))

with open('../../attention_is_all_you_need.txt', 'r') as f:
    text = f.read()

# Step 3: Create function to count tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


# Step 4: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=512,
    chunk_overlap=24,
    length_function=count_tokens,
)

chunks = text_splitter.create_documents([text])

embeddings = OpenAIEmbeddings()

# Create vector database
db = FAISS.from_documents(chunks, embeddings)

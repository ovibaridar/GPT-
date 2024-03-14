import os
import textract
from langchain.chains import ConversationalRetrievalChain
from transformers import GPT2TokenizerFast
from django.http import JsonResponse, HttpResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.language_models.base import BaseLanguageModel
from langchain.chains.question_answering import load_qa_chain

from . import Api_key  # Assuming you have a file named 'Api_key.py' containing your API key
import warnings

warnings.simplefilter("ignore")
from django.shortcuts import render

os.environ["OPENAI_API_KEY"] = Api_key.API_KEY

# Construct absolute path
file_path = 'Data_update/merged_file.pdf'
absolute_path = os.path.abspath(file_path)

loader = PyPDFLoader(absolute_path)

# Load and split pages
pages = loader.load_and_split()

# SKIP TO STEP 2 IF YOU'RE USING THIS METHOD
chunks = pages

# Advanced method - Split by chunk

# Step 1: Convert PDF to text
doc = textract.process(absolute_path)

# Step 2: Save to .txt and reopen (helps prevent issues)
with open('attention_is_all_you_need.txt', 'w', encoding='utf-8') as f:
    f.write(doc.decode('utf-8'))

with open('attention_is_all_you_need.txt', 'r', encoding='utf-8') as f:
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

# Define LLMChain class implementing BaseLanguageModel
from langchain.llms import OpenAI


class OpenAIBaseModel(BaseLanguageModel):
    def __init__(self, temperature, text):
        self.temperature = temperature

    def generate_prompt(self, text):
        # Implement prompt generation specific to OpenAI
        self.prompt = f"Input: {text}\n"
        return self.prompt


def chat_bot(request):
    if request.method == 'POST':
        user_input = request.POST.get('message')
        if not user_input:
            return JsonResponse({'error': 'No message provided'})

        query = user_input
        chat_history = []

        # Load the QA chain
        chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

        # Perform similarity search in the database
        docs = db.similarity_search(query)

        # Run the chain with the documents and query
        chain.run(input_documents=docs, question=query)

        # Create Conversational Retrieval Chain
        qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())

        # Get the result from Conversational Retrieval Chain
        result = qa({"question": query, "chat_history": chat_history})
        print(result)

        # Return the result in JSON format
        return JsonResponse({'response': result})


def home(request):
    return render(request, "index.html")

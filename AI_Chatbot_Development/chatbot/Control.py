from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
import main_controls

db = main_controls.db

from IPython.display import display
import ipywidgets as widgets


# Create conversation chain that uses our vectordb as retriver, this also allows for chat history management

def call(query):
    chat_history = []
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    docs = db.similarity_search(query)
    chain.run(input_documents=docs, question=query)
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())
    result = qa({"question": query, "chat_history": chat_history})

    return result["answer"]


que = "what is my  name ?"

print(call(que))

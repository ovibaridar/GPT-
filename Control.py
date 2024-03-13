from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
import main_control

db = main_control.db

while True:
    a = input("Enter Your Question : ")
    query = a
    docs = db.similarity_search(query)
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    docs = db.similarity_search(query)
    ans = chain.run(input_documents=docs, question=query)
    print(ans)

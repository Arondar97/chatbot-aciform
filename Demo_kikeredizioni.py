import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from HTML_Templates import css, bot_template, user_template
import os
from flask import Flask, request, jsonify

app = Flask(__name__)


def get_vectorstore(directory = 'v_testo_unico_articoli'):
    print('\n Load vectorstore')
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(directory, embeddings, allow_dangerous_deserialization=True)
    print('\n Vectorstore loaded')
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.3)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
    ) 
    return conversation_chain

@app.route('/demo_chatbot/json', methods=['POST'])
def chatbot_json():
    data = request.json
    json_data = data.get('question', '')
    load_dotenv()
    request = '''Sei un consulente di sicurezza sul lavoro. 
        In base alla domanda che ti viene fatta, elenca tutte le informazioni inerenti che possono rispondere a questa 
        e poi genera una risposta conclusiva. Alla fine di ogni risposta elenca sempre le pagine da cui hai preso le informazioni."
        La domanda è la seguente: ''' + json_data

    vectordb = get_vectorstore()
    conversation_chain = get_conversation_chain(vectordb)
    response = conversation_chain({"question": request, "chat_history": ''})
    print("\nsave response", response)
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
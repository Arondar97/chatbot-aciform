from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
#from langchain_openai import ChatOpenAI, OpenAIEmbeddings
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

# vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

directory = 'v_testo_unico'
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local(directory, embeddings, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever()


### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


### Answer question ###
system_prompt = (
    '''Sei un consulente di sicurezza sul lavoro. 
        In base alla domanda che ti viene fatta, elenca tutte le informazioni inerenti che possono rispondere a questa 
        e poi genera una risposta conclusiva. Alla fine di ogni risposta elenca sempre le pagine da cui hai preso le informazioni e gli articoli interenti."
        La domanda è la seguente: 
        \n\n
        {context}'''
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


### Create app callable by API
app = Flask(__name__)


@app.route('/')
def home():
    return "Benvenuto nel tuo Chat_bot personale!"

@app.route('/demo_chatbot/json', methods=['POST'])
def chatbot_json():
    data = request.json
    user_question = data.get('question', '')


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    response = conversational_rag_chain.invoke(
                {"input": user_question},
                config={
                "configurable": {"session_id": "1"}
                },
            )["answer"]
    return {'message': response}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

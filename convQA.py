from langchain.chains import (
    ConversationalRetrievalChain,
)
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import os
import chainlit as cl
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings

persist_directory = os.environ.get("PERSIST_DIRECTORY", "dburl")
embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")

vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
  
template = """
You are a knowledgeable assistant with comprehensive information about the University of North Texas (UNT). Utilize the provided conversation history and the current question to offer detailed, accurate, and helpful responses. Ensure your answers are informative and directly address the questions while integrating any relevant context from the conversation history.

CONTEXT:
{context}
- The University of North Texas (UNT) is a public research university in Denton, Texas.
- UNT offers 105 bachelor's, 88 master's, and 37 doctoral degree programs within its 14 colleges and schools.
- The university is known for its programs in education, engineering, music, information science, and business.

CONVERSATION HISTORY:
{chat_history}

CURRENT QUESTION: 
{question}

ANSWER:
"""


prompt = PromptTemplate(input_variables=["chat_history", "question", "context"], template=template)

condense_question_template = """
Given the conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that includes all necessary context for understanding. This rephrased question should be formulated in a way that it can be understood without needing to refer back to the conversation history, ensuring it contains all relevant details for accurate retrieval and response generation.

CONVERSATION HISTORY:
{chat_history}

FOLLOW-UP QUESTION:
{question}

REPHRASED STANDALONE QUESTION:
"""

condense_question_prompt = PromptTemplate.from_template(condense_question_template)

def load_model():
    llm = Ollama(
        model="gemma",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    return llm
    
@cl.on_chat_start
async def on_chat_start():
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    chain = ConversationalRetrievalChain.from_llm(
        load_model(),
        chain_type="stuff",
        retriever=retriever,
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": prompt},
        memory=memory,
        return_source_documents=True,
    )
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]
    # print("Source chunk: ",source_documents)
    
    text_elements = []

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"{source_doc.metadata['source']}"
            docname = source_name.replace("source_documents/", "")
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=docname)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources:\n" + "\n".join(source_names)
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()
import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain_community.vectorstores import Chroma, Weaviate, FAISS
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
import ollama
import os

# Load environment variables
load_dotenv()

# Set global model variables
model = os.getenv('MODEL', 'llama3.2')
chunk_size = int(os.getenv('CHUNK_SIZE', 500))
chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 50))

# set global model variables
#model = "mistral"
# model = "llama2"
#model = "llama3.2"

embeddings = OllamaEmbeddings(model=model)
retriever_cache = {}

# Function to load, split, and retrieve documents from URL or PDF
def load_and_retrieve_docs(source):
    if source in retriever_cache:
        print("source existing = ",source)
        return retriever_cache[source]

    if source.startswith("http"):
        loader = WebBaseLoader(web_paths=(source,), bs_kwargs=dict())
        
        docs = loader.load()

        #text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        splits = text_splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        #vectorstore = Weaviate.from_documents(documents=splits, embedding=embeddings)

        

    elif source.endswith(".pdf"):
        loader = PyMuPDFLoader(file_path=source)

        docs = loader.load()

        #text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)


    else:
        raise ValueError("Unsupported document source.")



    retriever = vectorstore.as_retriever()
    retriever_cache[source] = retriever

    return retriever

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function that defines the RAG chain for multiple URLs or PDFs
def rag_chain(source_url, source_pdf, question):
    print("source_url = ",source_url)
    print("source_pdf = ",source_pdf)
    print("retriever_cache = ", retriever_cache)

    responses = []

    if source_url:
            
        for src in source_url.split(","):
            retriever = load_and_retrieve_docs(src.strip())
            retrieved_docs = retriever.invoke(question)

            formatted_context = format_docs(retrieved_docs)
            formatted_prompt = f"Question: {question}\n\nContext: {formatted_context}"

            response = ollama.chat(model=model, messages=[{'role': 'user', 'content': formatted_prompt}])

            response_text = f"URL: {src}:\n\n{response['message']['content']}:\n"
            responses.append(response_text)

            print("response url: ",responses)



    if source_pdf:
        for src in source_pdf:

            retriever = load_and_retrieve_docs(src.strip())
            retrieved_docs = retriever.invoke(question)


            formatted_context = format_docs(retrieved_docs)
            formatted_prompt = f"Question: {question}\n\nContext: {formatted_context}"


            response = ollama.chat(model=model, messages=[{'role': 'user', 'content': formatted_prompt}])

            response_text = f"PDF: {src}:\n\n{response['message']['content']}:\n"
            responses.append(response_text)
    

            print("response pdf: ",responses)


    return '\n\n'.join(responses)

# Gradio interface setup with file upload or URL input for PDFs
iface = gr.Interface(
    fn=rag_chain,
    inputs=[
        gr.Textbox(label="Enter URL(s) separated by comma", type="text"),
        gr.File(label="Upload PDF File(s)", file_count="multiple"),
        "text"
    ],
    outputs="text",
    title="Retrieval-Augmented Generation (RAG) Chain Question Answering",
    description="Enter URL(s) separated by comma or upload a PDF file(s) to get answers from the RAG chain."
)

# Launch the Gradio app
iface.launch()
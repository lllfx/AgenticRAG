# encoding=utf-8
import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from googlesearch import search
from lingua import Language, LanguageDetectorBuilder
from trafilatura import fetch_url, extract

os.environ["HF_ENDPOINT"] = 'https://hf-mirror.com'
llm = ChatOpenAI(model='Qwen/Qwen2.5-7B-Instruct', api_key="xxx",
                 base_url="https://api.siliconflow.cn/v1")



def check_local_knowledge(query, context):
    """Router function to determine if we can answer from local knowledge"""
    prompt = '''Role: Question-Answering Assistant
Task: Determine whether the system can answer the user's question based on the provided text.
Instructions:
    - Analyze the text and identify if it contains the necessary information to answer the user's question.
    - Provide a clear and concise response indicating whether the system can answer the question or not.
    - Your response should include only a single word. Nothing else, no other text, information, header/footer. 
Output Format:
    - Answer: Yes/No
Study the below examples and based on that, respond to the last question. 
Examples:
    Input: 
        Text: The capital of France is Paris.
        User Question: What is the capital of France?
    Expected Output:
        Answer: Yes
    Input: 
        Text: The population of the United States is over 330 million.
        User Question: What is the population of China?
    Expected Output:
        Answer: No
    Input:
        User Question: {query}
        Text: {text}
'''
    formatted_prompt = prompt.format(text=context, query=query)
    response = llm.invoke(formatted_prompt)
    return response.content.strip().lower() == "yes"

def setup_vector_db(pdf_path):
    """Setup vector database from PDF"""
    # Load and chunk PDF
    # Create vector database
    embeddings = HuggingFaceEmbeddings(
        model_name="Dmeta-embedding-zh-small"
    )
    if os.path.exists("faiss_index"):
        vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        return vector_db
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(len(chunks))
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local("faiss_index")
    return vector_db


def get_local_content(vector_db, query):
    """Get content from vector database"""
    docs = vector_db.similarity_search(query, k=5)
    return " ".join([doc.page_content for doc in docs])


def generate_final_answer(context, query):
    """Generate final answer using LLM"""
    messages = [
        (
            "system",
            "You are a helpful assistant. Use the provided context to answer the query accurately.",
        ),
        ("system", f"Context: {context}"),
        ("human", query),
    ]
    response = llm.invoke(messages)
    return response.content


def get_main_text_trafilatura(url):
    downloaded = fetch_url(url)
    if downloaded is not None:
        result = extract(downloaded)
        return result
    else:
        return ""

def detect_language(text):
    languages = LanguageDetectorBuilder.from_all_languages().build()
    detected_language = languages.detect_language_of(text)
    language_name = detected_language.name
    if detected_language.name == "CHINESE":
        return "CHINESE"
    elif language_name == "ENGLISH":
        return "ENGLISH"
    else:
        return "Other"


def get_google_search_context(query):
    snippet = []
    for r in search(query, num_results=10, unique=True, advanced=True):
        text = get_main_text_trafilatura(r.url)
        if text == "": continue
        if detect_language(text) == "Other":
            continue
        snippet.append(f"title:{r.title}")
        snippet.append(f"content:{text}")
        snippet.append(f"link:{r.url}")
        snippet.append("")
    return "google search result\n" + "\n".join(snippet)
def process_query(query, vector_db, local_context):
    """Main function to process user query"""
    print(f"Processing query: {query}")

    # Step 1: Check if we can answer from local knowledge
    can_answer_locally = check_local_knowledge(query, local_context)
    print(f"Can answer locally: {can_answer_locally}")

    # Step 2: Get context either from local DB or web
    if can_answer_locally:
        context = get_local_content(vector_db, query)
        print("Retrieved context from local documents")
        print(context)
    else:
        print("Retrieved context from web scraping")
        context = get_google_search_context(query)
        print(context)
    # Step 3: Generate final answer
    answer = generate_final_answer(context, query)
    return answer


def main():
    # Setup
    pdf_path = "genai-principles.pdf"

    # Initialize vector database
    print("Setting up vector database...")
    vector_db = setup_vector_db(pdf_path)
    # Example usage
    query = "哈工大深圳研究生院怎么样"
    # Get initial context from PDF for routing
    local_context = get_local_content(vector_db, query)

    result = process_query(query, vector_db, local_context)
    print("\nFinal Answer:")
    print(result)


if __name__ == "__main__":
    main()

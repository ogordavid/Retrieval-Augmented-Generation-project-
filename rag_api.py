import os
from typing import List
from getpass import getpass
from dotenv import load_dotenv
import openai
import fastapi
os.environ.setdefault("USER_AGENT", "rag-pipeline-script")

load_dotenv()

# ---- Paste your OpenAI API key here ----
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# -----------------------------------------



from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# ---- Set what to load: a local file, or a webpage ----
FILE_PATH = "C:\\Users\\David\\Documents\\My resume-David Ogor.pdf"  # e.g. "/home/javis/GideonAkaniCS.pdf"
URLS = ["https://example.com"]            # used only if FILE_PATH is empty/None
# --------------------------------------------------------


class WebContentLoader:
    def __init__(self, urls: List[str]):
        self.urls = urls

    def load_content(self) -> List[Document]:
        try:
            documents = WebBaseLoader(self.urls).load()
            print(f"Successfully loaded content from {len(self.urls)} URLs")
            return documents
        except Exception as e:
            print(f"Error loading content: {str(e)}")
            return []


class LocalFileLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_content(self) -> List[Document]:
        try:
            loader = PyPDFLoader(self.file_path) if self.file_path.lower().endswith(".pdf") \
                else TextLoader(self.file_path, encoding="utf-8")
            documents = loader.load()
            print(f"Successfully loaded content from {self.file_path}")
            return documents
        except Exception as e:
            print(f"Error loading file: {str(e)}")
            return []


class DocumentChunker:
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def create_chunks(self, documents: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks


class PromptManager:
    @staticmethod
    def create_prompt_template() -> ChatPromptTemplate:
        system_prompt = (
            "You are an AI Assistant that follows instructions extremely well. "
            "Please be truthful and give direct answers. If the answer is not "
            "in the context, give your own answers.\n\nContext: {context}"
        )
        return ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])


def build_qa_chain():
    """Runs the full pipeline once: load -> chunk -> embed -> store -> retriever -> chain."""
    documents = LocalFileLoader(FILE_PATH).load_content() if FILE_PATH else WebContentLoader(URLS).load_content()
    chunks = DocumentChunker().create_chunks(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, max_tokens=512)
    prompt = PromptManager.create_prompt_template()
    question_answer_chain = create_stuff_documents_chain(model, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


# ---- Build the index once when the server starts ----
print("Building index, this may take a moment...")
qa_chain = build_qa_chain()
print("Ready.")

# ---- FastAPI app ----
app = FastAPI(title="RAG Query API")


class AskResponse(BaseModel):
    query: str
    answer: str


@app.get("/ask", response_model=AskResponse)
def ask(q: str):
    result = qa_chain.invoke({"input": q})
    return {"query": q, "answer": result["answer"]}


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ask your document</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; font-family: system-ui, sans-serif; background: #f4f5f7;
           height: 100vh; display: flex; justify-content: center; }
    .app { width: 100%; max-width: 700px; height: 100vh; display: flex;
           flex-direction: column; background: #fff;
           box-shadow: 0 0 20px rgba(0,0,0,.08); }
    header { padding: 16px 20px; border-bottom: 1px solid #eee; font-weight: 600; }
    #chat { flex: 1; overflow-y: auto; padding: 20px; display: flex;
            flex-direction: column; gap: 12px; }
    .msg { max-width: 80%; padding: 10px 14px; border-radius: 16px;
           line-height: 1.5; white-space: pre-wrap; }
    .user { align-self: flex-end; background: #2563eb; color: #fff;
            border-bottom-right-radius: 4px; }
    .bot { align-self: flex-start; background: #eef0f3; color: #111;
           border-bottom-left-radius: 4px; }
    .thinking { color: #888; font-style: italic; }
    .input-row { display: flex; gap: 8px; padding: 16px; border-top: 1px solid #eee; }
    #q { flex: 1; padding: 12px 14px; border: 1px solid #ddd;
         border-radius: 24px; font-size: 15px; outline: none; }
    #q:focus { border-color: #2563eb; }
    button { padding: 0 20px; border: none; border-radius: 24px;
             background: #2563eb; color: #fff; font-size: 15px; cursor: pointer; }
    button:disabled { background: #9db7f0; cursor: not-allowed; }
  </style>
</head>
<body>
  <div class="app">
    <header>Ask your document</header>
    <div id="chat">
      <div class="msg bot">Hi! Ask me anything about the document.</div>
    </div>
    <div class="input-row">
      <input id="q" type="text" placeholder="Type your question..." autocomplete="off" />
      <button id="send">Send</button>
    </div>
  </div>

  <script>
    const chat = document.getElementById('chat');
    const input = document.getElementById('q');
    const btn = document.getElementById('send');

    function addMsg(text, cls) {
      const div = document.createElement('div');
      div.className = 'msg ' + cls;
      div.textContent = text;
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
      return div;
    }

    async function ask() {
      const q = input.value.trim();
      if (!q) return;
      addMsg(q, 'user');
      input.value = '';
      btn.disabled = true;
      const reply = addMsg('Thinking...', 'bot thinking');
      try {
        const res = await fetch('/ask?q=' + encodeURIComponent(q));
        if (!res.ok) throw new Error('Server error');
        const data = await res.json();
        reply.className = 'msg bot';
        reply.textContent = data.answer;
      } catch (err) {
        reply.className = 'msg bot';
        reply.textContent = 'Something went wrong. Try again.';
      }
      btn.disabled = false;
      input.focus();
    }

    btn.addEventListener('click', ask);
    input.addEventListener('keydown', function (e) { if (e.key === 'Enter') ask(); });
  </script>
</body>
</html>
"""

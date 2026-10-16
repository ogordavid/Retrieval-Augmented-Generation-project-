\# RAG Resume API



A question-answering API built with FastAPI, LangChain, Chroma and OpenAI. It loads a PDF, splits it into chunks, embeds them, and answers questions about it. It includes a simple chat-style web UI.



\## How it works

1\. Load the document (PDF or text file)

2\. Split it into small chunks

3\. Convert the chunks to embeddings and store them in Chroma

4\. For each question, retrieve the most relevant chunks and pass them to gpt-4o-mini to generate the answer



\## Setup

1\. Clone the repo and create a virtual environment

2\. Install the dependencies:

```

&#x20;  pip install -r requirements.txt

```

3\. Copy `.env.example` to `.env` and fill in your OpenAI API key and the path to your PDF



\## Run

```

uvicorn rag\_api:app --reload

```

Then open http://127.0.0.1:8000 for the chat UI, or http://127.0.0.1:8000/docs for the API docs.



\## Tech stack

FastAPI, LangChain, Chroma, OpenAI (text-embedding-3-small, gpt-4o-mini)


Here is the **full directory structure** and **all necessary files** for your **Flask + React-based chatbot UI** with **Jira, Confluence, AWS Bedrock, and ChromaDB integration**.

---

## **📂 Project Directory Structure**
```
chatbot-project/
│── backend/                  # Flask API Backend
│   │── app.py                # Flask application (API)
│   │── requirements.txt       # Dependencies for Flask
│   └── pdf_dir/               # Stores downloaded Confluence PDFs
│── frontend/                  # React Frontend
│   │── public/                # Static assets
│   │   │── logo.png           # Company logo
│   │   │── banner.png         # Banner image
│   │   └── chatbot-logo.png   # Chatbot icon
│   │── src/                   # React source files
│   │   │── App.js             # Main React component
│   │   │── index.js           # React entry point
│   │   │── components/        # UI Components
│   │   │   ├── Header.js      # Header Component
│   │   │   ├── ChatSection.js # Chat UI
│   │   │   ├── History.js     # Chat History Component
│   │   │   ├── FloatingButtons.js # Jira & Confluence floating buttons
│   │   │── assets/            # UI Assets (images, icons)
│   │   └── styles/            # Styling (CSS, Tailwind)
│   │       ├── App.css        # Global styles
│   │── tailwind.config.js     # Tailwind Configuration
│   │── package.json           # React dependencies
│   └── README.md              # Instructions
```

---

## **📌 Backend (Flask API)**
📁 **File: `backend/app.py`**
```python
from flask import Flask, request, jsonify
import os
import json
import boto3
import re
import pdfplumber
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from atlassian import Jira, Confluence

app = Flask(__name__)

# AWS Bedrock client
brt = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# Confluence & Jira Credentials
CONFLUENCE_URL = "https://confluence.org.com"
JIRA_URL = "https://jira.org.com"
USERNAME = "gbudfa"
PASSWORD = "your-password"
API_TOKEN = "your-jira-api-token"

# Initialize Jira & Confluence
jira = Jira(url=JIRA_URL, username=USERNAME, password=API_TOKEN, verify_ssl=False)
confluence = Confluence(url=CONFLUENCE_URL, username=USERNAME, password=PASSWORD, verify_ssl=False)

# Embedding function
class TitanEmbeddingFunction:
    def __init__(self, model_id, region="us-east-1"):
        self.model_id = model_id
        self.bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            response = brt.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({"inputText": text})
            )
            embedding = json.loads(response["body"].read())["embedding"]
            embeddings.append(embedding)
        return embeddings

# ChromaDB
embedding_function = TitanEmbeddingFunction(model_id="amazon.titan-embed-text-v2:0")
client = chromadb.PersistentClient(path="./knowledge_base")
collection = client.get_or_create_collection(name="my_collection", embedding_function=embedding_function)

# Process PDFs & Store in ChromaDB
@app.route("/process_pdfs", methods=["POST"])
def process_pdfs():
    pdf_dir = "./pdf_dir"
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            process_large_pdf(pdf_path)
    return jsonify({"message": "PDFs processed successfully."})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("query", "")
    if not user_query:
        return jsonify({"error": "Query cannot be empty"}), 400

    model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    response, confluence_links, other_pdf_sources = query_chromadb_and_generate_response(
        user_query, collection, embedding_function, model_id
    )

    return jsonify({"response": response, "confluence_links": confluence_links, "other_pdf_sources": list(other_pdf_sources)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
```

📁 **File: `backend/requirements.txt`**
```
flask
boto3
chromadb
pdfplumber
atlassian
PyPDF2
```
💡 Install backend dependencies:
```bash
pip install -r backend/requirements.txt
```

---

## **📌 Frontend (React)**
📁 **File: `frontend/src/App.js`**
```jsx
import React, { useState } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import { FaJira, FaConfluence } from "react-icons/fa";
import Header from "./components/Header";
import ChatSection from "./components/ChatSection";
import History from "./components/History";
import FloatingButtons from "./components/FloatingButtons";

export default function App() {
  const [conversation, setConversation] = useState([]);

  return (
    <div className="min-h-screen">
      <Header />
      <div className="max-w-6xl mx-auto mt-6 grid grid-cols-3 gap-4 px-6">
        <ChatSection setConversation={setConversation} />
        <History conversation={conversation} />
      </div>
      <FloatingButtons />
    </div>
  );
}
```

📁 **File: `frontend/src/components/Header.js`**
```jsx
export default function Header() {
  return (
    <div className="w-full bg-blue-600 flex justify-between items-center py-4 px-6 shadow-md">
      <img src="/logo.png" alt="Company Logo" className="h-12" />
      <h1 className="text-white text-2xl font-bold">AI Cloud Chatbot</h1>
      <img src="/banner.png" alt="Banner" className="h-14 rounded-md shadow-md" />
    </div>
  );
}
```

📁 **File: `frontend/src/components/ChatSection.js`**
```jsx
import React, { useState } from "react";
import axios from "axios";

export default function ChatSection({ setConversation }) {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");

  const sendQuery = async () => {
    if (!query.trim()) return;
    try {
      const { data } = await axios.post("http://localhost:5000/chat", { query });
      setResponse(data.response);
      setConversation((prev) => [...prev, { question: query, answer: data.response }]);
      setQuery("");
    } catch (error) {
      console.error("Error fetching response:", error);
    }
  };

  return (
    <div className="bg-white p-6 rounded-xl shadow-lg col-span-2">
      <input
        type="text"
        className="w-full border rounded-lg p-3"
        placeholder="Type your question..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && sendQuery()}
      />
      <button onClick={sendQuery} className="mt-4 bg-blue-600 text-white p-2 rounded-lg">
        Ask AI
      </button>
      {response && <div className="mt-4 bg-gray-100 p-4 rounded-lg">{response}</div>}
    </div>
  );
}
```

---

## **🚀 Running the Project**
1️⃣ **Start Flask API:**
```bash
cd backend
python app.py
```
2️⃣ **Start React Frontend:**
```bash
cd frontend
npm start
```
3️⃣ **Visit:** `http://localhost:3000`

✅ **Your chatbot UI is now live!** 🚀

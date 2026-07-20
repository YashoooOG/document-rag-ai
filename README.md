
# 🚀 Introduction

Document RAG AI allows users to upload PDF documents and ask questions about their contents. Instead of sending the entire document to an LLM, the application:


# ✨ Features

- 📄 Upload multiple PDF files
- 🔍 Automatic text extraction
- ✂️ Intelligent text chunking
- 🧠 Hugging Face embeddings
- ⚡ FAISS vector search
- 🤖 Google Gemini powered responses
- 💬 Chat interface built with Streamlit
- 🔄 Reset conversation anytime

---

# 🛠 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Streamlit | Web Application |
| PyMuPDF | PDF Text Extraction |
| LangChain | RAG Pipeline |
| Hugging Face Embeddings | Text Embeddings |
| FAISS | Vector Database |
| Google Gemini | Large Language Model |
| python-dotenv | Environment Variable Management |

---

# 📁 Project Structure

```text
document-rag-ai/
│
├── app.py
├── testllm.py
├── README.md
├── requirement.txt
├── LICENSE
└── .env
```

---

# ⚙️ How It Works

```
Upload PDF
      │
      ▼
Extract Text (PyMuPDF)
      │
      ▼
Split into Chunks
      │
      ▼
Generate Embeddings
      │
      ▼
Store in FAISS
      │
      ▼
User Question
      │
      ▼
Retrieve Relevant Chunks
      │
      ▼
Gemini Generates Answer
```

---

# 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YashoooOG/document-rag-ai.git
cd document-rag-ai
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirement.txt
```

### 5. Create a `.env` File

Create a `.env` file in the project root and add your Google Gemini API key.

```env
GOOGLE_API_KEY=YOUR_API_KEY
```

---

# ▶ Running the Application

Run the Streamlit application using:

```bash
streamlit run app.py
```

The application will automatically open in your browser.

If it doesn't open automatically, visit:

```
http://localhost:8501
```

---

# 💬 Using the Application

1. Launch the application.
2. Upload one or more PDF documents.
3. Click **Process**.
4. Wait for indexing to complete.
5. Ask questions related to your uploaded documents.
6. Reset the conversation anytime using the **Reset** button.

---

# 📌 Requirements

- Python 3.10+
- Google Gemini API Key
- Internet connection (for Gemini API)

---


# 👨‍💻 Author

**Tanishq Verma**

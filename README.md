## ⚡ Quick Start

### 1. Clone the Repository
```
git clone https://github.com/FHJibon/PDF-Extractor
```


### 2. Setup Backend Environment
```
cd PDF-Extractor
```
```
python -m venv .venv
```
```
.venv\Scripts\activate
```
```
pip install -r requirements.txt
```

### 3. Configure OpenAI API Key
- Create a `.env` file
```
OPENAI_API_KEY=
```

### 4. Start Backend Server
```
uvicorn app.main:app --reload
```

[![Visit API Docs](https://img.shields.io/badge/Visit-API%20Docs-0EA5E9?style=for-the-badge)](http://127.0.0.1:8000/docs)

---
**Note:** This project requires an OpenAI account and API key. The backend uses the OpenAI Python SDK for document parsing and invoice extraction.
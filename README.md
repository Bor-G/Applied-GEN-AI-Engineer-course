# **CV Analysis System Documentation**

## **Project Overview**
The CV Analysis System is a comprehensive application developed in two phases, combining advanced CV processing capabilities with intelligent conversation features. The system uses AI-powered analysis for CV processing and provides both structured data retrieval and natural language interaction. 
   
The **_screenshots** folder contains screenshots of the screen that illustrate this document.

---

## **Part 1: Core CV Analysis Implementation**

### 🔧 **Technologies Used**
* **Python 3.8+**: Core programming language
* **Streamlit**: Web interface framework
* **PostgreSQL with pgvector**: Vector database for embeddings storage
* **Google Vertex AI**: LLM services for text analysis
* **PyPDF2**: PDF processing
* **Pandas**: Data manipulation
* **Langchain**: Framework for LLM operations

### 🏗️ **Core Components Implementation**

#### **CV Processing Engine**
```python
# DocumentProcessor class (document_processor.py)
class DocumentProcessor:
    """Handles CV processing and analysis"""
    def __init__(self):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
```
* PDF text extraction using PyPDF2
* Recursive text splitting for chunking
* LLM-based summary generation
* Skill extraction with fallback mechanisms

#### **Vector Database Integration**
```sql
-- Database schema example
CREATE TABLE IF NOT EXISTS candidates (
    id SERIAL PRIMARY KEY,
    identifier TEXT,
    current_position TEXT,
    experience_years INTEGER,
    key_skills TEXT,
    cv_text TEXT,
    summary TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cv_embeddings (
    id SERIAL PRIMARY KEY,
    candidate_id INTEGER REFERENCES candidates(id),
    chunk_text TEXT,
    embedding vector(768),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```
* PostgreSQL with pgvector extension
* Schema design for CV data and embeddings
* Efficient vector similarity search
* Transaction handling and connection management

#### **Embeddings Management**
```python
# Example embedding generation
embeddings = embeddings_handler.generate_embeddings(text_chunks)
db.store_embeddings(candidate_id, embeddings)
```
* Integration with Google's Vertex AI embeddings
* Vector similarity calculations
* Bulk embedding generation
* Efficient embedding storage and retrieval

**Steps Implemented**
   - **CV Upload and Processing**:
     - Extracted CV text from PDFs using `PyPDF2`.
     - Split text into manageable chunks using `langchain.text_splitter`.
   - **Embedding Generation**:
     - Created vector embeddings with `VertexAIEmbeddings`.
   - **Database Storage**:
     - Designed a PostgreSQL schema to store CV data and embeddings.
     - Used `psycopg2` to interact with the database.
   - **Candidate Summary**:
     - Generated summaries and key skills using LLM prompts.
   - **UI Implementation**:
     - Built Streamlit tabs for uploading CVs, viewing candidates, and querying.

**Outcome**
   - A web-based application allowing users to upload and analyze CVs and retrieve insights through an interactive UI.

---

## **Part 2: Advanced Agent Integration**

### 🚀 **New Technologies Added**
* **LangChain and LangGraph**: For building complex conversation flows
* **DuckDuckGo Search**: Web search integration
* **LangGraph**: State management for translation workflow
* **Additional LangChain components**: For agent implementation

### 🔄 **New Components Implementation**

#### **ReAct Agent System**
```python
# CVAnalysisAgent interaction example
agent = CVAnalysisAgent(llm, db, embeddings_handler)
response = agent.run("List all current roles that are mentioned in the CV files. List them in descending order of number, giving the role name and number in a table.")
```
* Tool-based architecture
* Query routing and processing
* Temporal context awareness
* Multi-tool coordination

#### **Translation Agent**
```python
# Translation request example
translation = translation_agent.run(
    "Translate from English to Ukrainian: Yesterday all my troubles seemed so far away."
)
```
* LangGraph-based workflow
* Language detection
* State management
* Error handling

#### **Enhanced Search Integration**
```python
# Enhanced search example
search_tool = EnhancedDuckDuckGoSearchRun()
results = search_tool.run("What did Donald Trump say about Ukraine after his election as President of the United States?")
```
* Web search integration
* Result filtering
* Temporal context addition
* Source attribution

**Steps Implemented**
   - **Retrieval Tool**:
     - Implemented a `CVRetrievalTool` to fetch relevant CV data from the database.
     - Enabled similarity search using vector embeddings.
   - **Additional Tools**:
     - Developed a `CandidateAnalysisTool` for detailed candidate insights.
     - Integrated `DuckDuckGoSearchRun` for general knowledge queries.
   - **ReAct Agent**:
     - Combined tools into a LangChain-based ReAct agent.
     - Configured prompts to handle multi-tool reasoning and user interactions dynamically.
   - **Translation Capabilities**:
     - Added a `TranslationAgent` to handle multilingual translation requests.

**Outcome**
   - An advanced chatbot capable of answering CV-specific and general questions while offering translation services.

---

## 🛠️ **Installation and Setup**

### **Prerequisites**
1. Python 3.8+ environment
2. PostgreSQL database with pgvector extension
3. Google Cloud project with Vertex AI access
4. Required Python packages from `requirements.txt`

### **Environment Setup**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### **Configuration**
Create a `.env` file:
```plaintext
GOOGLE_CLOUD_PROJECT=<Your_Google_Cloud_Project_ID>
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-service-account-key.json
DB_HOST=localhost
DB_PORT=5432
DB_NAME=cv_analysis
DB_USER=postgres
DB_PASSWORD=<Your_Database_Password>
```

Ensure that PostgreSQL and the required extensions (`pgvector`) are set up.

The Google Cloud service account must have all the necessary permissions, and the corresponding JSON key file must be downloaded and the file path set to the `GOOGLE_APPLICATION_CREDENTIALS` variable in the `.env` file.

### **Database Setup**
The application handles automatic table creation on startup:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS candidates (...);
CREATE TABLE IF NOT EXISTS cv_embeddings (...);
```

---

## **Starting the Application**
1. Navigate to the project directory in the terminal.
2. Run the application using:
   ```bash
   streamlit run app.py
   ```
3. Open the application in your browser using the URL displayed in the terminal, usually `http://localhost:8501`.

---

## 📚 **Usage Guide**

### **CV Management**

#### **1. Uploading CVs**
1. Navigate to the sidebar
2. Click "Browse files"
3. Select PDF files for processing
4. System automatically extracts and analyzes content

#### **2. Viewing Candidates**
* Switch to "Candidates" tab
* Select candidate from dropdown
* View comprehensive profile:
  - Basic information
  - Current role
  - Experience
  - Key skills
  - Generated summary
  - Full CV view

### **AI Assistant Features**

#### **1. CV Analysis Queries**
```plaintext
"What are main key skills?"
"Show me all Information Technology Managers"
"List all current roles"
```

#### **2. Translation Capabilities**
```plaintext
"Translate from English to Ukrainian: Hello, world!"
"Translate this to Ukrainian: I am a software developer"
"Translate from Ukrainian to English: Доброго дня"
```

#### **3. General Knowledge Queries**
```plaintext
"What are the best practices for CV writing?"
"Tell me about recent developments in AI"
```

### **Advanced Features**

#### 🔍 **Role Statistics**
* View distribution of roles
* Get experience level breakdowns
* Analyze skill frequency

#### 🌐 **Multi-language Support**
* Bidirectional translation
* Support for multiple language pairs
* Preservation of formatting and style

#### ⏱️ **Temporal Analysis**
* Current events awareness
* Historical context preservation
* Future reference handling

---

## ⚠️ **Error Handling**

The system handles various errors:
* PDF processing issues
* Database connection problems
* Translation errors
* API limitations
* Query processing failures

Each error includes:
```python
try:
    # Operation code
except Exception as e:
    log_error(e)
    return user_friendly_message
```

---

## 🏛️ **Architecture Benefits**

### **1. Modularity**
* Separate components for different functionalities
* Easy maintenance and updates
* Clear dependency management

### **2. Scalability**
* Vector database for efficient searching
* Asynchronous processing capabilities
* Resource-efficient design

### **3. Extensibility**
* Tool-based architecture
* Plugin system for new capabilities
* Flexible agent integration

---

## ⚡ **Limitations and Considerations**

### **1. PDF Processing**
* Some complex PDF formats may not process correctly
* Image-based PDFs require OCR (not implemented)

### **2. Language Support**
* Translation quality depends on source/target language pair
* Some languages may have limited support

---

## 🛠️ **Troubleshooting**

### **1. Database Connection Issues**
- Ensure PostgreSQL is running, and the `.env` file contains the correct credentials.

### **2. Google Cloud Errors**
- Verify that your Google Cloud Project is correctly set up, the `.env` file contains the correct credentials, and all necessary APIs (e.g., Vertex AI) are enabled.

### **3. PDF Processing Errors**
- Ensure the uploaded files are valid PDFs. Corrupted or unsupported PDFs might not be processed.

### **4. Chatbot Not Responding**
- Ensure embeddings are generated for the uploaded CVs. If no embeddings are stored, the chatbot cannot answer CV-related questions.

---

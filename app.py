"""Main Streamlit application for CV analysis and translation."""
import streamlit as st
from src.database import DatabaseHandler
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingsHandler
from src.react_agent import CVAnalysisAgent
from src.translation_agent import TranslationAgent
from src.config import *
from langchain_google_vertexai import ChatVertexAI
import pandas as pd
import json
from typing import Tuple


def init_llm() -> ChatVertexAI:
    """Initialize the LLM model."""
    return ChatVertexAI(
        project=GOOGLE_CLOUD_PROJECT,
        model="gemini-1.5-pro-001",
        temperature=0.7,
        max_output_tokens=2048
    )


def init_components() -> Tuple[DatabaseHandler, DocumentProcessor, EmbeddingsHandler,
                               ChatVertexAI, CVAnalysisAgent, TranslationAgent]:
    """Initialize all necessary components for the application."""
    # Initialize database handler
    db = DatabaseHandler(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

    # Initialize LLM
    llm = init_llm()

    # Initialize document processor
    doc_processor = DocumentProcessor(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        llm=llm
    )

    # Initialize embeddings handler
    embeddings_handler = EmbeddingsHandler(
        project_id=GOOGLE_CLOUD_PROJECT
    )

    # Initialize agents
    cv_agent = CVAnalysisAgent(llm, db, embeddings_handler)
    translation_agent = TranslationAgent()

    # Ensure database tables are created
    db.init_tables()

    return db, doc_processor, embeddings_handler, llm, cv_agent, translation_agent


def get_candidate_profiles(db: DatabaseHandler) -> pd.DataFrame:
    """Retrieve all candidate profiles from database."""
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, identifier, current_position, experience_years, 
                       key_skills, cv_text, summary
                FROM candidates
                ORDER BY created_at DESC;
            """)
            columns = ['id', 'identifier', 'current_position', 'experience_years',
                       'key_skills', 'cv_text', 'summary']
            rows = cur.fetchall()

            data = []
            for row in rows:
                row_data = list(row)
                try:
                    if row_data[4] and isinstance(row_data[4], str):
                        row_data[4] = json.loads(row_data[4])
                    elif not row_data[4]:
                        row_data[4] = []
                except json.JSONDecodeError:
                    row_data[4] = []
                data.append(row_data)

            return pd.DataFrame(data, columns=columns)


def process_cv_file(uploaded_file, db, doc_processor, embeddings_handler):
    """Process a single CV file."""
    try:
        cv_text = doc_processor.extract_text_from_pdf(uploaded_file)
        candidate_info = doc_processor.extract_candidate_info(
            cv_text,
            uploaded_file.name
        )

        # Check if candidate already exists
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id FROM candidates 
                    WHERE identifier = %s
                """, (candidate_info["identifier"],))
                existing_candidate = cur.fetchone()

        if existing_candidate:
            return f"CV for {candidate_info['identifier']} already exists in the database."

        # Store candidate information
        candidate_id = db.store_candidate(
            candidate_info["identifier"],
            candidate_info["current_position"],
            candidate_info["experience_years"],
            candidate_info["key_skills"],
            cv_text,
            candidate_info["summary"]
        )

        # Generate and store embeddings
        chunks = doc_processor.split_text(cv_text)
        chunk_embeddings = embeddings_handler.generate_embeddings(chunks)
        db.store_embeddings(candidate_id, chunk_embeddings)

        return f"Successfully processed CV for {candidate_info['identifier']}"
    except Exception as e:
        return f"Error processing CV {uploaded_file.name}: {str(e)}"


def init_session_state():
    """Initialize all session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None


def main():
    """Main application function."""
    st.set_page_config(
        page_title="CV Analysis System",
        page_icon="📄",
        layout="wide"
    )

    init_session_state()

    st.title("CV Analysis System")

    try:
        # Initialize all components
        db, doc_processor, embeddings_handler, llm, cv_agent, translation_agent = init_components()
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        st.stop()

    # Sidebar for uploading CVs
    with st.sidebar:
        st.header("Upload CVs")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True
        )

        if uploaded_files:
            with st.spinner("Processing CVs..."):
                for uploaded_file in uploaded_files:
                    if uploaded_file.name not in st.session_state.processed_files:
                        result = process_cv_file(
                            uploaded_file,
                            db,
                            doc_processor,
                            embeddings_handler
                        )
                        st.write(result)
                        st.session_state.processed_files.add(uploaded_file.name)

    # Main content area with tabs
    tab1, tab2 = st.tabs(["Candidates", "AI Assistant"])

    # Tab 1: Candidate Profiles
    with tab1:
        st.header("Candidate Profiles")
        try:
            candidates_df = get_candidate_profiles(db)
            total_candidates = len(candidates_df)
            st.write(f"Total candidates in database: **{total_candidates}**")

            if candidates_df.empty:
                st.info("No candidates in the database. Please upload some CVs.")
            else:
                selected_candidate = st.selectbox(
                    "Select a candidate to view details:",
                    options=candidates_df['identifier'].tolist(),
                    key="candidate_selector"
                )

                if selected_candidate:
                    candidate_info = candidates_df[
                        candidates_df['identifier'] == selected_candidate
                    ].iloc[0]

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Basic Information")
                        st.write(f"**Title:** {candidate_info['identifier']}")
                        st.write(f"**Current Role:** {candidate_info['current_position']}")
                        st.write(f"**Years of Experience:** {candidate_info['experience_years']}")

                    with col2:
                        st.subheader("Summary")
                        st.write(candidate_info['summary'])

                        if isinstance(candidate_info['key_skills'], list) and candidate_info['key_skills']:
                            st.subheader("Key Skills")
                            st.write(", ".join(candidate_info['key_skills']))

                    with st.expander("View Full CV"):
                        st.text(candidate_info['cv_text'])

        except Exception as e:
            st.error(f"Error loading candidate profiles: {str(e)}")

    # Tab 2: AI Assistant
    with tab2:
        st.header("AI Assistant")

        st.markdown("""
        This AI Assistant can help you with:
        - Searching and analyzing CVs in the database
        - Providing detailed candidate analysis
        - Answering general questions
        - Translating text between languages (e.g., English to Ukrainian)

        Example translation requests:
        - "Translate to Ukrainian: Hello, how are you?"
        - "Translate from English to Ukrainian: Good morning"
        - "Translate this text to Ukrainian: I am a software developer"

        Feel free to ask any questions!
        """)

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask me anything about the candidates, request translations, or get career advice"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response using appropriate agent
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Route to appropriate agent
                        if translation_agent.is_translation_query(prompt):
                            response = translation_agent.run(prompt)
                        else:
                            response = cv_agent.run(prompt)

                        st.markdown(response)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )
                    except Exception as e:
                        error_message = f"Error generating response: {str(e)}"
                        st.error(error_message)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_message}
                        )


if __name__ == "__main__":
    main()

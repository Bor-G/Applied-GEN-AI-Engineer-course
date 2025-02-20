"""Database operations for CV analysis application."""
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
import json


class DatabaseHandler:
    """Handles database operations for CV analysis."""

    def __init__(self, host, port, dbname, user, password):
        """
        Initialize the database connection.

        Args:
            host (str): Database host.
            port (str): Database port.
            dbname (str): Database name.
            user (str): Database user.
            password (str): Database password.
        """
        self.conn_params = {
            "host": host,
            "port": port,
            "dbname": dbname,
            "user": user,
            "password": password
        }
        self.connection = None

    def connect(self):
        """
        Establish a database connection and register the vector extension.

        Returns:
            psycopg2.connection: Database connection object.
        """
        self.connection = psycopg2.connect(**self.conn_params)
        register_vector(self.connection)
        return self.connection

    def init_tables(self):
        """Initialize the database tables if they do not exist."""
        with self.connect() as conn:
            with conn.cursor() as cur:
                # Create extensions if not exists
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # Create tables with updated column names
                cur.execute("""
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
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS cv_embeddings (
                        id SERIAL PRIMARY KEY,
                        candidate_id INTEGER REFERENCES candidates(id),
                        chunk_text TEXT,
                        embedding vector(768),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.commit()

    def store_candidate(self, identifier, current_position, experience_years, key_skills, cv_text, summary):
        """Store candidate information in the database."""
        print(f"Storing Candidate - Identifier: {identifier}, Current Role: {current_position}")
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                   INSERT INTO candidates (identifier, current_position, experience_years, key_skills, cv_text, summary)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   RETURNING id;
                """, (identifier, current_position, experience_years, json.dumps(key_skills), cv_text, summary))
                return cur.fetchone()[0]

    def store_embeddings(self, candidate_id, chunk_embeddings):
        """Store document chunk embeddings in the database."""
        with self.connect() as conn:
            with conn.cursor() as cur:
                data = [(candidate_id, chunk, embedding)
                        for chunk, embedding in chunk_embeddings]
                execute_values(cur, """
                    INSERT INTO cv_embeddings (candidate_id, chunk_text, embedding)
                    VALUES %s
                """, data)
                conn.commit()

    def get_embeddings(self):
        """Retrieve all embeddings from database with their corresponding chunks."""
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT e.id, e.candidate_id, e.chunk_text, e.embedding,
                           c.identifier as candidate_name
                    FROM cv_embeddings e
                    JOIN candidates c ON e.candidate_id = c.id
                    ORDER BY e.id
                """)
                return cur.fetchall()

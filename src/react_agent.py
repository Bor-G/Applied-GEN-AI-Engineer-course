"""ReAct agent implementation with improved temporal awareness and query handling."""
from typing import Dict, Any
from datetime import datetime
import json
from pydantic import Field
from langchain.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate


class EnhancedDuckDuckGoSearchRun(BaseTool):
    """Enhanced search tool with temporal awareness."""
    name: str = Field(default="web_search")
    description: str = Field(
        default=(
            "Use this tool to search for factual information from the web. "
            "Only use this tool for general knowledge queries or current events. "
            "Returns search results with source attribution and temporal context."
        )
    )
    search_engine: Any = Field(default_factory=DuckDuckGoSearchRun)
    current_date: datetime = Field(default_factory=lambda: datetime(2024, 11, 24))

    class Config:
        arbitrary_types_allowed = True

    def __init__(self):
        """Initialize the enhanced search tool."""
        super().__init__()
        self.search_engine = DuckDuckGoSearchRun()
        self.current_date = datetime(2024, 11, 24)

    def _run(self, query: str) -> str:
        try:
            # Add date context to the query
            date_context = f"as of {self.current_date.strftime('%B %Y')}"
            contextualized_query = f"{query} {date_context}"

            results = self.search_engine.run(contextualized_query)
            processed_results = self._process_results(results, query)

            return json.dumps({
                "type": "search_results",
                "query": query,
                "results": processed_results,
                "temporal_context": {
                    "current_date": self.current_date.strftime("%Y-%m-%d"),
                    "search_date": date_context
                },
                "source": "DuckDuckGo Search"
            })
        except Exception as e:
            return json.dumps({
                "type": "error",
                "message": f"Search failed: {str(e)}"
            })

    @staticmethod
    def _process_results(results: str, original_query: str) -> str:
        """Process and filter search results with temporal awareness."""
        response_parts = []

        # Split into sentences and analyze each
        sentences = results.split('. ')
        for sentence in sentences:
            # Skip sentences that are clearly speculative
            if any(word in sentence.lower() for word in [
                'would', 'could', 'might', 'may', 'possible', 'potentially',
                'expected to', 'predicted', 'forecast', 'projected'
            ]):
                continue

            # Skip sentences about future events
            if any(word in sentence.lower() for word in [
                'will be', 'going to', 'plans to', 'intends to',
                'upcoming', 'future', 'next year'
            ]):
                continue

            response_parts.append(sentence)

        if not response_parts:
            return "No factual information available for this query."

        return ". ".join(response_parts) + "."


class CVRetrievalTool(BaseTool):
    """Tool for retrieving CV information from the vector database."""
    name: str = "cv_retriever"
    description: str = Field(
        default=(
            "Use this tool to search through CV database and retrieve relevant information. "
            "For role statistics, use with input 'get_role_statistics'. "
            "For candidate search, provide a search query. "
            "Only use this tool for CV-related queries."
        )
    )
    db: Any = Field(default=None)
    embeddings: Any = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, db_handler, embeddings_handler):
        """Initialize with database and embeddings handlers."""
        super().__init__()
        self.db = db_handler
        self.embeddings = embeddings_handler

    def _run(self, query: str) -> str:
        try:
            # Handle role statistics query first
            if query.strip().lower() == 'get_role_statistics':
                return self._get_role_statistics()

            # If not a statistics query, proceed with embedding-based search
            query_embeddings = self.embeddings.generate_embeddings([query])
            query_embedding = query_embeddings[0][1]

            stored_data = self.db.get_embeddings()
            if not stored_data:
                return json.dumps({
                    "type": "error",
                    "message": "No CV data available in the database."
                })

            stored_embeddings = [row[3] for row in stored_data]
            chunks_with_metadata = [(row[2], row[4]) for row in stored_data]

            similar_indices = self.embeddings.similarity_search(
                query_embedding,
                stored_embeddings
            )

            relevant_results = [
                {
                    "text": chunks_with_metadata[idx][0],
                    "candidate": chunks_with_metadata[idx][1],
                    "relevance": score
                }
                for idx, score in similar_indices
                if score > 0.7
            ]

            if not relevant_results:
                return json.dumps({
                    "type": "error",
                    "message": "No relevant information found in the CV database."
                })

            return json.dumps({
                "type": "cv_search_results",
                "results": relevant_results[:3]
            })

        except Exception as e:
            return json.dumps({
                "type": "error",
                "message": f"Error retrieving CV information: {str(e)}"
            })

    def _get_role_statistics(self) -> str:
        """Get role statistics directly from the database."""
        try:
            with self.db.connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            current_position as role,
                            COUNT(*) as count
                        FROM candidates 
                        WHERE current_position IS NOT NULL 
                        AND current_position != 'Role not specified'
                        GROUP BY current_position
                        ORDER BY count DESC, current_position ASC
                    """)
                    roles = cur.fetchall()

                    if not roles:
                        return json.dumps({
                            "type": "role_statistics",
                            "data": []
                        })

                    return json.dumps({
                        "type": "role_statistics",
                        "data": [{"role": role, "count": count} for role, count in roles]
                    })

        except Exception as e:
            print(f"Database error in _get_role_statistics: {str(e)}")  # Add debugging
            return json.dumps({
                "type": "error",
                "message": f"Error retrieving role statistics: {str(e)}"
            })


class CandidateAnalysisTool(BaseTool):
    """Tool for analyzing candidate qualifications and generating insights."""
    name: str = "candidate_analyzer"
    description: str = Field(
        default=(
            "Use this tool to analyze candidate qualifications and generate insights. "
            "Returns structured data about candidates that needs to be interpreted appropriately."
        )
    )
    db: Any = Field(default=None)
    llm: Any = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, db_handler, llm):
        """Initialize with database handler and LLM."""
        super().__init__()
        self.db = db_handler
        self.llm = llm

    def _run(self, query: str) -> str:
        try:
            with self.db.connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT identifier, current_position, experience_years, 
                               key_skills, summary
                        FROM candidates
                        WHERE identifier LIKE %s
                    """, (f"%{query}%",))
                    results = cur.fetchone()

            if not results:
                return json.dumps({
                    "type": "error",
                    "message": f"No candidate found matching '{query}'"
                })

            return json.dumps({
                "type": "candidate_info",
                "data": {
                    "identifier": results[0],
                    "current_position": results[1],
                    "experience_years": results[2],
                    "key_skills": json.loads(results[3]) if results[3] else [],
                    "summary": results[4]
                }
            })

        except Exception as e:
            return json.dumps({
                "type": "error",
                "message": f"Error analyzing candidate information: {str(e)}"
            })


class CVAnalysisAgent:
    """ReAct agent for CV analysis system."""

    def __init__(self, llm, db_handler, embeddings_handler):
        self.llm = llm
        self.db_handler = db_handler
        self.current_date = datetime(2024, 11, 24)

        # Initialize tools
        self.tools = [
            CVRetrievalTool(db_handler, embeddings_handler),
            CandidateAnalysisTool(db_handler, llm),
            EnhancedDuckDuckGoSearchRun()
        ]

        # Create the React prompt template
        prompt = PromptTemplate(
            input_variables=["input", "agent_scratchpad", "tool_names", "tools", "current_date"],
            template="""You are a professional assistant specializing in CV analysis and information retrieval.
            You have access to both a CV database and web search capabilities.
            Current date context: {current_date}
            
            IMPORTANT GUIDELINES:
            1. Always consider the temporal context of queries
            2. For CV-related queries, use cv_retriever or candidate_analyzer
            3. For current events or general knowledge, use web_search
            4. For role statistics:
               - Use cv_retriever with input 'get_role_statistics'
               - Format the results as a proper markdown table
               - Include total counts and percentages if available
            
            When handling role statistics results:
            1. If the data is empty, explain that no roles are found in the database
            2. If data is available, format it as a markdown table with headers "Role | Count"
            3. Sort roles by count in descending order
            4. Include a summary of total roles and candidates
            
            You have access to the following tools:
            
            {tools}
            
            To use a tool, use the following format:
            Thought: Do I need to use a tool? Yes
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the tool
            Observation: the result of the action
            
            When you have a final answer, use this format:
            Thought: I know what to say
            Final Answer: [your response here]
            
            Begin!
            
            Question: {input}
            {agent_scratchpad}"""
        )

        self.agent_executor = AgentExecutor(
            agent=create_react_agent(
                llm=llm,
                tools=self.tools,
                prompt=prompt
            ),
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

    def _analyze_temporal_context(self, query: str) -> Dict[str, Any]:
        """Analyze the temporal context of a query."""
        # Extract any dates or temporal references
        current_year = self.current_date.year
        current_month = self.current_date.month
        query_lower = query.lower()

        temporal_info = {
            "is_temporal_query": False,
            "is_future": False,
            "referenced_date": None,
            "temporal_context": "present"
        }

        # Check for year references
        if str(current_year) in query:
            temporal_info["is_temporal_query"] = True
            # Parse month/date context if available
            if "january" in query_lower or "jan" in query_lower:
                temporal_info["referenced_date"] = datetime(current_year, 1, 1)
            elif "november" in query_lower or "nov" in query_lower:
                temporal_info["referenced_date"] = datetime(current_year, 11, 1)
            # Add more month checks as needed
            else:
                temporal_info["referenced_date"] = datetime(current_year, 1, 1)

        # If we have a referenced date, determine if it's future
        if temporal_info["referenced_date"]:
            temporal_info["is_future"] = temporal_info["referenced_date"] > self.current_date
            temporal_info["temporal_context"] = (
                "future" if temporal_info["is_future"]
                else "past" if temporal_info["referenced_date"] < self.current_date
                else "present"
            )

        return temporal_info

    def run(self, query: str) -> str:
        try:
            # Clean and preprocess the query
            query = query.strip()

            # Analyze temporal context
            temporal_info = self._analyze_temporal_context(query)

            # Handle different temporal contexts
            if temporal_info["is_temporal_query"]:
                if temporal_info["temporal_context"] == "future":
                    return (
                        "I cannot provide information about events after "
                        f"{self.current_date.strftime('%B %d, %Y')}. "
                        "I can only provide information about verified past events "
                        "and current situations based on available data."
                    )
                elif temporal_info["temporal_context"] == "past":
                    # Let the agent handle past events
                    pass
                else:
                    # Add context about current date for present queries
                    query = f"{query} (as of {self.current_date.strftime('%B %d, %Y')})"

            # All queries go through the agent for consistent handling
            response = self.agent_executor.invoke({
                "input": query,
                "tools": "\n".join(f"{tool.name}: {tool.description}" for tool in self.tools),
                "tool_names": ", ".join(tool.name for tool in self.tools),
                "current_date": self.current_date.strftime("%B %d, %Y")
            })

            return response["output"]

        except Exception as e:
            print(f"Error details: {str(e)}")
            return (
                "I encountered an error processing your query. "
                "Please try rephrasing your question."
            )

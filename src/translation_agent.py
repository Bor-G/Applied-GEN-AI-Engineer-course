"""Translation agent implementation using LangGraph."""
from typing import Annotated, Dict, TypedDict, Union, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_vertexai import ChatVertexAI
import re


class AgentState(TypedDict):
    """State definition for translation agent."""
    messages: List[BaseMessage]
    current_language: str
    target_language: str
    translation_done: bool


def create_translation_llm():
    """Create LLM instance for translation."""
    return ChatVertexAI(
        model="gemini-1.5-pro-001",
        temperature=0.1,  # Low temperature for more accurate translations
        max_output_tokens=1024,
        top_p=0.95
    )


def detect_language_and_direction(state: AgentState) -> AgentState:
    """Detect source language and translation direction."""
    messages = state["messages"]
    if not messages:
        state["translation_done"] = True
        return state

    last_message = messages[-1].content.lower()

    # Check if it's a translation request
    translation_patterns = [
        r'translate\s+(?:from\s+)?(\w+)\s+(?:to|into)\s+(\w+)',
        r'(?:translate|convert)\s+(?:this|following|text)?\s*(?:to|into)\s+(\w+)',
        r'(?:translate|convert)\s+(?:to|into)\s+(\w+)'
    ]

    for pattern in translation_patterns:
        match = re.search(pattern, last_message)
        if match:
            # Update state with detected languages
            groups = match.groups()
            if len(groups) == 2:
                state["current_language"] = groups[0].lower()
                state["target_language"] = groups[1].lower()
            else:
                state["current_language"] = "english"  # Default source language
                state["target_language"] = groups[0].lower()
            return state

    # If no translation pattern is found
    state["translation_done"] = True
    return state


def translate_text(state: AgentState) -> AgentState:
    """Perform the translation using LLM."""
    llm = create_translation_llm()
    last_message = state["messages"][-1].content

    # Extract the actual text to translate
    text_to_translate = re.sub(
        r'(?:translate|convert)\s+(?:from\s+\w+\s+)?(?:to|into)\s+\w+\s*[:|\n]?\s*',
        '',
        last_message,
        flags=re.IGNORECASE
    ).strip()

    if not text_to_translate:
        text_to_translate = last_message

    # Create translation prompt
    system_prompt = f"""You are a professional translator.
    Translate the following text from {state['current_language']} to {state['target_language']}.
    Maintain the original meaning, tone, and formatting.
    For poetry and literary texts, preserve the artistic style and rhythm where possible.
    Only provide the translation, without any additional explanations or notes.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text_to_translate}
    ]

    try:
        # Get translation
        response = llm.invoke(messages)
        translation = response.content.strip()

        # Update state
        state["messages"].append(AIMessage(content=translation))
        state["translation_done"] = True

        return state

    except Exception as e:
        error_message = f"Translation error: {str(e)}"
        state["messages"].append(AIMessage(content=error_message))
        state["translation_done"] = True
        return state


def handle_non_translation(state: AgentState) -> AgentState:
    """Handle non-translation queries."""
    state["messages"].append(
        AIMessage(content=(
            "I am a translation assistant. I can help you translate text between "
            "different languages. Please provide a translation request in the format: "
            "'translate [text] to [language]' or 'translate from [source language] "
            "to [target language]: [text]'"
        ))
    )
    state["translation_done"] = True
    return state


def should_continue(state: AgentState) -> Union[str, bool]:
    """Determine if the workflow should continue."""
    if state["translation_done"]:
        return END
    if state.get("current_language") and state.get("target_language"):
        return "translate"
    return "handle_non_translation"


class TranslationAgent:
    """LangGraph-based translation agent."""

    def __init__(self):
        """Initialize the translation agent."""
        # Create workflow graph
        self.workflow = StateGraph(AgentState)

        # Add nodes
        self.workflow.add_node("detect_language", detect_language_and_direction)
        self.workflow.add_node("translate", translate_text)
        self.workflow.add_node("handle_non_translation", handle_non_translation)

        # Add conditional edges
        self.workflow.add_conditional_edges(
            "detect_language",
            should_continue
        )

        # Add edge from translate to end
        self.workflow.add_edge("translate", END)
        self.workflow.add_edge("handle_non_translation", END)

        # Set entry point
        self.workflow.set_entry_point("detect_language")

        # Compile the graph
        self.chain = self.workflow.compile()

    def run(self, query: str) -> str:
        """Run the translation agent on a query."""
        # Initialize state
        state = AgentState(
            messages=[HumanMessage(content=query)],
            current_language="",
            target_language="",
            translation_done=False
        )

        try:
            # Run the workflow
            final_state = self.chain.invoke(state)

            # Return the last message
            if final_state["messages"]:
                return final_state["messages"][-1].content
            return "No response generated."
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return f"An error occurred during translation: {str(e)}"

    @staticmethod
    def is_translation_query(query: str) -> bool:
        """Check if the query is a translation request."""
        translation_patterns = [
            r'translate\s+(?:from\s+)?\w+\s+(?:to|into)\s+\w+',
            r'(?:translate|convert)\s+(?:this|following|text)?\s*(?:to|into)\s+\w+',
            r'translate\s+(?:to|into)\s+\w+'
        ]

        return any(re.search(pattern, query.lower()) for pattern in translation_patterns)

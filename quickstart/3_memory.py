import json
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()


# Define the State type for the graph
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Initialize MemorySaver for persistent memory across conversations
memory = MemorySaver()

# Initialize the search tool
tool = TavilySearchResults(max_results=2)
tools = [tool]

# Initialize the LLM
LOCAL_LLM = 'llama3.1:8b'
llm = ChatOpenAI(
    api_key="ollama",
    model=LOCAL_LLM,
    base_url="http://localhost:11434/v1",
)
llm_with_tools = llm.bind_tools(tools)


# Define the chatbot function
def chatbot(state: State):
    """
    Process the current state and generate a response using the LLM with tools.

    Args:
        state (State): The current state containing messages.

    Returns:
        dict: A dictionary with the 'messages' key containing the LLM's response.
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Build the graph
def build_graph():
    """
    Create and compile the StateGraph for the chatbot with tool integration and memory.

    Returns:
        StateGraph: The compiled graph for the chatbot.
    """
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)

    tool_node = ToolNode(tools=[tool])
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    return graph_builder.compile(checkpointer=memory)


# Function to display the graph
def display_graph(graph):
    """
    Attempt to display the graph in ASCII format.

    Args:
        graph (StateGraph): The compiled graph to display.
    """
    try:
        print(graph.get_graph().draw_ascii())
    except Exception as e:
        print(f"Error displaying the graph: {e}")


# Function to process user input and display assistant responses
def process_input(graph, user_input: str, thread_id: str):
    """
    Process user input through the graph and display assistant responses.

    Args:
        graph (StateGraph): The compiled graph to use.
        user_input (str): The user's input message.
        thread_id (str): The unique identifier for the conversation thread.
    """
    config = {"configurable": {"thread_id": thread_id}}
    events = graph.stream(
        {"messages": [("user", user_input)]}, config, stream_mode="values"
    )
    for event in events:
        event["messages"][-1].pretty_print()


# Main function to demonstrate the chatbot with memory
def main():
    """
    Main function to demonstrate the chatbot with memory functionality.
    """
    graph = build_graph()
    display_graph(graph)

    print("\nDemonstrating chatbot with memory:")

    # First interaction
    print("\n--- Interaction 1 (Thread 1) ---")
    process_input(graph, "Hi there! My name is Will.", "1")

    # Second interaction (same thread)
    print("\n--- Interaction 2 (Thread 1) ---")
    process_input(graph, "Remember my name?", "1")

    # Third interaction (different thread)
    print("\n--- Interaction 3 (Thread 2) ---")
    process_input(graph, "Remember my name?", "2")

    # Display snapshot of Thread 1
    print("\n--- Snapshot of Thread 1 ---")
    config = {"configurable": {"thread_id": "1"}}
    snapshot = graph.get_state(config)
    print('Snapshot:')
    print(snapshot)
    print('Next Snapshot:')
    print(snapshot.next)


if __name__ == "__main__":
    main()

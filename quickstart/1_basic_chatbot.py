from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Define the State type for the graph
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


# Initialize the LLM
LOCAL_LLM = 'llama3.1:8b'
llm = ChatOllama(model=LOCAL_LLM, temperature=0)


# Define the chatbot function
def chatbot(state: State):
    """
    Process the current state and generate a response using the LLM.

    Args:
        state (State): The current state containing messages.

    Returns:
        dict: A dictionary with the 'messages' key containing the LLM's response.
    """
    return {"messages": [llm.invoke(state["messages"])]}


# Build the graph
def build_graph():
    """
    Create and compile the StateGraph for the chatbot.

    Returns:
        StateGraph: The compiled graph for the chatbot.
    """
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    return graph_builder.compile()


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


# Function to stream graph updates
def stream_graph_updates(graph, user_input: str):
    """
    Stream updates from the graph based on user input.

    Args:
        graph (StateGraph): The compiled graph to use.
        user_input (str): The user's input message.
    """
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


# Main function to run the chatbot
def main():
    """
    Main function to run the chatbot interactively.
    """
    graph = build_graph()
    display_graph(graph)

    print("Chatbot initialized. Type 'quit', 'exit', or 'q' to end the conversation.")

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(graph, user_input)
        except EOFError:
            # Fallback if input() is not available (e.g., in non-interactive environments)
            print("Non-interactive mode detected. Using default input.")
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(graph, user_input)
            break


if __name__ == "__main__":
    main()

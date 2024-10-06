import json
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage

# Load environment variables from .env file
load_dotenv()


# Define the State type for the graph
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Initialize the search tool
tool = TavilySearchResults(max_results=2)
tools = [tool]

# Initialize the LLM
LOCAL_LLM = 'llama3.1:8b'

# Pretend Ollama mode as OpenAI (since bind_tools is not implemented for ChatOllama)
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


# Define the BasicToolNode class
class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


# Define the routing function
def route_tools(state: State):
    """
    Route to the ToolNode if the last message has tool calls. Otherwise, route to the end.

    Args:
        state (State): The current state.

    Returns:
        str: "tools" if tool calls are present, END otherwise.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


# Build the graph
def build_graph():
    """
    Create and compile the StateGraph for the chatbot with tool integration.

    Returns:
        StateGraph: The compiled graph for the chatbot.
    """
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)

    tool_node = BasicToolNode(tools=[tool])
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        {"tools": "tools", END: END},
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

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
            if value["messages"] and value["messages"][-1].content.strip():
                print("Assistant:", value["messages"][-1].content.strip())


# Main function to run the chatbot
def main():
    """
    Main function to run the chatbot interactively with tool integration.
    """
    graph = build_graph()
    display_graph(graph)

    print("Chatbot initialized with tool integration. Type 'quit', 'exit', or 'q' to end the conversation.")

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

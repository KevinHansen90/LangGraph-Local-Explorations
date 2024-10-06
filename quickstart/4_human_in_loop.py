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

# Initialize MemorySaver for persistent memory across conversations
memory = MemorySaver()


# Define the State type for the graph
class State(TypedDict):
	messages: Annotated[list, add_messages]


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
	Create and compile the StateGraph for the chatbot with tool integration,
	memory, and human-in-the-loop functionality.

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

	return graph_builder.compile(
		checkpointer=memory,
		interrupt_before=["tools"],  # This allows for human intervention before tool use
	)


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


# Function to process user input and display responses
def process_input(graph, user_input: str, config: dict):
	"""
	Process user input through the graph and display responses.

	Args:
		graph (StateGraph): The compiled graph to use.
		user_input (str): The user's input message.
		config (dict): Configuration for the graph, including thread_id.
	"""
	events = graph.stream(
		{"messages": [("user", user_input)]}, config, stream_mode="values"
	)
	for event in events:
		if "messages" in event:
			event["messages"][-1].pretty_print()


# Main function to demonstrate the chatbot with human-in-the-loop functionality
def main():
	"""
	Main function to demonstrate the chatbot with interruption and resumption.
	"""
	graph = build_graph()
	display_graph(graph)

	print("\nDemonstrating chatbot with interruption and resumption:")

	user_input = "I'm learning LangGraph. Could you do some research on it for me?"
	config = {"configurable": {"thread_id": "1"}}

	print("\n--- Initial User Input ---")
	events = graph.stream(
		{"messages": [("user", user_input)]}, config, stream_mode="values"
	)
	for event in events:
		if "messages" in event:
			event["messages"][-1].pretty_print()

	print("\n--- Current State (At Interruption Point) ---")
	snapshot = graph.get_state(config)
	existing_message = snapshot.values["messages"][-1]
	existing_message.pretty_print()

	print("\n--- Resuming Execution ---")
	events = graph.stream(None, config, stream_mode="values")
	for event in events:
		if "messages" in event:
			event["messages"][-1].pretty_print()


if __name__ == "__main__":
	main()

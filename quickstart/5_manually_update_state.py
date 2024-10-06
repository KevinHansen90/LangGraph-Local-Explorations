from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage

# Load environment variables from .env file
load_dotenv()

# Initialize MemorySaver for persistent memory across conversations
memory = MemorySaver()


# Define the State type for the graph
class State(TypedDict):
	messages: Annotated[list, add_messages]


# Initialize the graph builder, tool, and LLM
graph_builder = StateGraph(State)
tool = TavilySearchResults(max_results=2)
tools = [tool]
local_llm = 'llama3.1:8b'
llm = ChatOpenAI(
	api_key="ollama",
	model=local_llm,
	base_url="http://localhost:11434/v1",
)
llm_with_tools = llm.bind_tools(tools)


# Define the chatbot function
def chatbot(state: State):
	return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Build the graph
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Compile the graph with interruption before tool use
graph = graph_builder.compile(
	checkpointer=memory,
	interrupt_before=["tools"],
)


def display_graph(graph):
	try:
		print(graph.get_graph().draw_ascii())
	except Exception as e:
		print(f"Error displaying the graph: {e}")


def main():
	display_graph(graph)

	# Initial user input and processing
	user_input = "I'm learning LangGraph. Could you do some research on it for me?"
	config = {"configurable": {"thread_id": "1"}}

	print("\n--- Initial User Input and Response ---")
	events = graph.stream({"messages": [("user", user_input)]}, config, stream_mode="values")
	for event in events:
		if "messages" in event:
			event["messages"][-1].pretty_print()

	# Display current state
	snapshot = graph.get_state(config)
	existing_message = snapshot.values["messages"][-1]
	print("\n--- Current State ---")
	existing_message.pretty_print()

	# Manually update the state with a predefined answer
	print("\n--- Manually Updating State ---")
	answer = "LangGraph is a library for building stateful, multi-actor applications with LLMs."
	new_messages = [
		ToolMessage(content=answer, tool_call_id=existing_message.tool_calls[0]["id"]),
		AIMessage(content=answer),
	]
	new_messages[-1].pretty_print()
	graph.update_state(config, {"messages": new_messages})

	print("\n--- Last 2 Messages After Update ---")
	print(graph.get_state(config).values["messages"][-2:])

	# Update state as if the chatbot node just ran
	print("\n--- Updating State as Chatbot Node ---")
	graph.update_state(
		config,
		{"messages": [AIMessage(content="I'm an AI expert!")]},
		as_node="chatbot",
	)

	# Display updated state
	snapshot = graph.get_state(config)
	print("\n--- Last 3 Messages and Next Node ---")
	print(snapshot.values["messages"][-3:])
	print("Next node:", snapshot.next)

	# Demonstrate overwriting existing messages in a new thread
	print("\n--- Overwriting Existing Messages (New Thread) ---")
	config_2 = {"configurable": {"thread_id": "2"}}
	events = graph.stream({"messages": [("user", user_input)]}, config_2, stream_mode="values")
	for event in events:
		if "messages" in event:
			event["messages"][-1].pretty_print()

	# Modify the existing message
	snapshot = graph.get_state(config_2)
	existing_message = snapshot.values["messages"][-1]
	print("\n--- Original Message ---")
	print("Message ID:", existing_message.id)
	print(existing_message.tool_calls[0])

	new_tool_call = existing_message.tool_calls[0].copy()
	new_tool_call["args"]["query"] = "LangGraph human-in-the-loop workflow"
	new_message = AIMessage(
		content=existing_message.content,
		tool_calls=[new_tool_call],
		id=existing_message.id,  # Important for replacing the message
	)

	print("\n--- Updated Message ---")
	print(new_message.tool_calls[0])
	print("Message ID:", new_message.id)
	graph.update_state(config_2, {"messages": [new_message]})

	print("\n--- Updated Tool Calls ---")
	print(graph.get_state(config_2).values["messages"][-1].tool_calls)

	# Continue execution with modified state
	print("\n--- Continuing Execution with Modified State ---")
	events = graph.stream(None, config_2, stream_mode="values")
	for event in events:
		if "messages" in event:
			event["messages"][-1].pretty_print()

	# Final user interaction to check memory
	print("\n--- Final User Interaction ---")
	events = graph.stream(
		{"messages": ("user", "Remember what I'm learning about?")},
		config_2,
		stream_mode="values",
	)
	for event in events:
		if "messages" in event:
			event["messages"][-1].pretty_print()


if __name__ == "__main__":
	main()

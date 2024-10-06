from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Define the State type with a new 'ask_human' flag
class State(TypedDict):
	messages: Annotated[list, add_messages]
	ask_human: bool


# Define a custom function for requesting human assistance
class RequestAssistance(BaseModel):
	"""Escalate the conversation to an expert. Use this if you are unable to assist directly or if the user requires support beyond your permissions."""
	request: str


# Initialize tools and LLM
tool = TavilySearchResults(max_results=2)
tools = [tool]
LOCAL_LLM = 'llama3.1:8b'
llm = ChatOpenAI(
	api_key="ollama",
	model=LOCAL_LLM,
	base_url="http://localhost:11434/v1",
)
llm_with_tools = llm.bind_tools(tools + [RequestAssistance])


# Define the chatbot function
def chatbot(state: State):
	response = llm_with_tools.invoke(state["messages"])
	ask_human = False
	if (
			response.tool_calls
			and response.tool_calls[0]["name"] == RequestAssistance.__name__
	):
		ask_human = True
	return {"messages": [response], "ask_human": ask_human}


# Define the human node function
def create_response(response: str, ai_message: AIMessage):
	return ToolMessage(
		content=response,
		tool_call_id=ai_message.tool_calls[0]["id"],
	)


def human_node(state: State):
	new_messages = []
	if not isinstance(state["messages"][-1], ToolMessage):
		new_messages.append(
			create_response("No response from human.", state["messages"][-1])
		)
	return {
		"messages": new_messages,
		"ask_human": False,
	}


# Define the node selection function
def select_next_node(state: State):
	if state["ask_human"]:
		return "human"
	return tools_condition(state)


# Build the graph
def build_graph():
	graph_builder = StateGraph(State)
	graph_builder.add_node("chatbot", chatbot)
	graph_builder.add_node("tools", ToolNode(tools=[tool]))
	graph_builder.add_node("human", human_node)

	graph_builder.add_conditional_edges(
		"chatbot",
		select_next_node,
		{"human": "human", "tools": "tools", END: END},
	)
	graph_builder.add_edge("tools", "chatbot")
	graph_builder.add_edge("human", "chatbot")
	graph_builder.add_edge(START, "chatbot")

	memory = MemorySaver()
	return graph_builder.compile(
		checkpointer=memory,
		interrupt_before=["human"],
	)


# Function to display the graph
def display_graph(graph):
	try:
		print(graph.get_graph().draw_ascii())
	except Exception as e:
		print(f"Error displaying the graph: {e}")


# Function to process user input and display responses
def process_input(graph, user_input: str, config: dict):
	events = graph.stream(
		{"messages": [("user", user_input)], "ask_human": False},
		config,
		stream_mode="values"
	)
	for event in events:
		if "messages" in event:
			event["messages"][-1].pretty_print()
		if event.get("ask_human", False):
			print("Human assistance requested. Interrupting execution.")
			return True
	return False


# Main function to run the chatbot
def main():
	graph = build_graph()
	display_graph(graph)

	print("\nCustomized State Chatbot with Human-in-the-Loop Capability")
	print("Type 'quit' to exit the conversation.")

	config = {"configurable": {"thread_id": "1"}}
	while True:
		user_input = input("\nUser: ")
		if user_input.lower() == 'quit':
			print("Exiting the conversation. Goodbye!")
			break

		human_requested = process_input(graph, user_input, config)

		if human_requested:
			human_response = input("Human Expert (type 'skip' to let AI continue): ")
			if human_response.lower() != 'skip':
				graph.update_state(
					config,
					{
						"messages": [create_response(human_response, graph.get_state(config).values["messages"][-1])],
						"ask_human": False
					}
				)
			events = graph.stream(None, config, stream_mode="values")
			for event in events:
				if "messages" in event:
					event["messages"][-1].pretty_print()


if __name__ == "__main__":
	main()

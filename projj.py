import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.redis import RedisSaver

load_dotenv()

REDIS_URL = "redis://localhost:6380"


class GraphState(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatGroq(model="openai/gpt-oss-120b")


def chat_node(state: GraphState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(GraphState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)


with RedisSaver.from_conn_string(REDIS_URL) as checkpointer:
    checkpointer.setup()

    graph = builder.compile(checkpointer=checkpointer)

    def chat_agent(user_message: str, thread_id: str) -> str:
        result = graph.invoke(
            {"messages": [("user", user_message)]},
            config={"configurable": {"thread_id": thread_id}}
        )
        return result["messages"][-1].content


    if __name__ == "__main__":
        thread_id = "redis-session-1"

        print("Redis-backed chat agent ready. Type 'exit' to quit.")

        while True:
            user_input = input("\nUser: ").strip()

            if user_input.lower() in ("exit", "quit", "q"):
                break

            print("Agent:", chat_agent(user_input, thread_id))
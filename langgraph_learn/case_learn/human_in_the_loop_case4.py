from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command

class State(dict):
    some_text: str

def human_node(state: State):
    print("state start", state)
    value = interrupt(
        {
            "text_to_revise": state["some_text"]
        }
    )
    print("state end", value)
    return {"some_text": value}

graph_builder = StateGraph(State)
graph_builder.add_node("human", human_node)
graph_builder.set_entry_point("human")
graph_builder.set_finish_point("human")

checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

# Run the graph until the interrupt is hit.
config = {"configurable": {"thread_id": "some_id"}}
result = graph.invoke({"some_text": "original text"}, config=config)

print(result['__interrupt__'])
# > [
# >    Interrupt(
# >       value={'text_to_revise': 'original text'},
# >       resumable=True,
# >       ns=['human_node:6ce9e64f-edef-fe5d-f7dc-511fa9526960']
# >    )
# > ]

print("here")
print(graph.invoke(Command(resume="Edited text"), config=config))
# > {'some_text': 'Edited text'}
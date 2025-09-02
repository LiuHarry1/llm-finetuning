
def print_state_history(graph, thread_id):

    history = list(graph.get_state_history({"configurable": {"thread_id": thread_id}}))
    history.reverse()  # 反转，确保从最早到最新

    for i, step in enumerate(history, start=1):
        foo_val = step.values.get("foo")
        bar_val = step.values.get("bar")
        print(f"Step {i}: foo={foo_val} | bar={bar_val}")

    print("\n=== checkpint 历史 每一个步骤里的状态快照 ===")
    for i, step in enumerate(history, start=1):
        print(step)
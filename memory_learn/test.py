from llama_index.core.memory import Memory
import asyncio


async def main():
    memory = Memory.from_defaults(
        session_id="my_session",
        token_limit=20,  # Normally you would set this to be closer to the LLM context window (i.e. 75,000, etc.)
        token_flush_size=10,
        chat_history_token_ratio=0.7,
    )

    from llama_index.core.llms import ChatMessage

    # Simulate a long conversation
    for i in range(100):
        await memory.aput_messages(
            [
                ChatMessage(role="user", content="Hello, world!"),
                ChatMessage(role="assistant", content="Hello, world to you too!"),
                ChatMessage(role="user", content="What is the capital of France?"),
                ChatMessage(
                    role="assistant", content="The capital of France is Paris."
                ),
            ]
        )

    current_chat_history = await memory.aget()
    for msg in current_chat_history:
        print(msg)

    all_messages = await memory.aget_all()
    print(len(all_messages))

    await memory.areset()

    all_messages = await memory.aget_all()
    print(len(all_messages))

if __name__ == "__main__":
    asyncio.run(main())


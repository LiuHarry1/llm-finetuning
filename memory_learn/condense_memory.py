import tiktoken
from pydantic import Field
from typing import List, Optional, Any
from llama_index.core.llms import ChatMessage, TextBlock
from llama_index.core.memory import Memory, BaseMemoryBlock
import asyncio


class CondensedMemoryBlock(BaseMemoryBlock[str]):
    current_memory: List[str] = Field(default_factory=list)
    token_limit: int = Field(default=50000)
    tokenizer: tiktoken.Encoding = tiktoken.encoding_for_model(
        "gpt-4o"
    )  # all openai models use 4o tokenizer these days

    async def _aget(
        self, messages: Optional[List[ChatMessage]] = None, **block_kwargs: Any
    ) -> str:
        """Return the current memory block contents."""
        return "\n".join(self.current_memory)

    async def _aput(self, messages: List[ChatMessage]) -> None:
        """Push messages into the memory block. (Only handles text content)"""
        # construct a string for each message
        for message in messages:
            text_contents = "\n".join(
                block.text
                for block in message.blocks
                if isinstance(block, TextBlock)
            )
            memory_str = f""

            if text_contents:
                memory_str += f"\n{text_contents}"

            # include additional kwargs, like tool calls, when needed
            # filter out injected session_id
            kwargs = {
                key: val
                for key, val in message.additional_kwargs.items()
                if key != "session_id"
            }
            if kwargs:
                memory_str += f"\n({kwargs})"

            memory_str += "\n"
            self.current_memory.append(memory_str)

        # ensure this memory block doesn't get too large
        message_length = sum(
            len(self.tokenizer.encode(message))
            for message in self.current_memory
        )
        while message_length > self.token_limit:
            self.current_memory = self.current_memory[1:]
            message_length = sum(
                len(self.tokenizer.encode(message))
                for message in self.current_memory
            )

async def main():
    block = CondensedMemoryBlock(name="condensed_memory")

    memory = Memory.from_defaults(
        session_id="test-mem-01",
        token_limit=30,
        token_flush_size=10,
        async_database_uri="sqlite+aiosqlite:///:memory:",
        memory_blocks=[block],
        insert_method="user",
        # Prevent the short-term chat history from containing too many turns!
        # This limit will effectively mean that the short-term memory is always flushed
        chat_history_token_ratio=0.0001,
    )

    initial_messages = [
        ChatMessage(role="user", content="Hello! My name is Logan"),
        ChatMessage(role="assistant", content="Hello! How can I help you?"),
        ChatMessage(role="user", content="What is the capital of France?"),
        ChatMessage(role="assistant", content="The capital of France is Paris"),
    ]

    await memory.aput_messages(initial_messages)

    await memory.aput_messages(
        [ChatMessage(role="user", content="What was my name again?")]
    )

    chat_history = await memory.aget()

    for message in chat_history:
        print(message.role)
        print(message.content)
        print()

if __name__ == "__main__":
    asyncio.run(main())
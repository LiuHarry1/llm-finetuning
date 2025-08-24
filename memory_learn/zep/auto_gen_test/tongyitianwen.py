from typing import Sequence, AsyncGenerator, Union, Any, Mapping, Optional, Literal
from autogen_core.models import (
    ChatCompletionClient,
    LLMMessage,
    CreateResult,
    RequestUsage,
    ModelCapabilities,
    ModelInfo,
    ModelFamily,
)
from langchain_community.chat_models import ChatTongyi

class TongyiChatCompletionClient(ChatCompletionClient):
    def __init__(self, model: str, api_key: str):
        self._llm = ChatTongyi(model=model, api_key=api_key)
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._actual_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._model_info = ModelInfo(
            name=model,
            family=ModelFamily.UNKNOWN,
            vision=False,
        )

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = [],
        tool_choice: Union[Any, Literal["auto", "required", "none"]] = "auto",
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token=None,
    ) -> CreateResult:
        # 拼接上下文
        text = "\n".join([m.content for m in messages])
        resp = await self._llm.apredict(text)

        result = CreateResult(
            content=resp,
            usage=RequestUsage(prompt_tokens=0, completion_tokens=0)
        )
        return result

    def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = [],
        tool_choice: Union[Any, Literal["auto", "required", "none"]] = "auto",
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token=None,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        # 简单实现：直接调用 create，不支持流式
        async def gen():
            result = await self.create(messages, tools=tools, tool_choice=tool_choice,
                                       json_output=json_output, extra_create_args=extra_create_args,
                                       cancellation_token=cancellation_token)
            yield result.content
            yield result
        return gen()

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return self._actual_usage

    def total_usage(self) -> RequestUsage:
        return self._total_usage

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = []) -> int:
        return 0  # 简化：不计数

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = []) -> int:
        return 999999  # 简化：无限

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()

    @property
    def model_info(self) -> ModelInfo:
        return self._model_info

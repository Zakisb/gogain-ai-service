"""NutritionAgent definition – handles diet planning and meal adjustments."""

import logging
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional

from langchain.agents import AgentExecutor, create_openai_tools_agent
from app.core.langchain_compat import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.agents.langchain_utils import (
    convert_dict_messages_to_langchain,
)
from app.core.config import DEFAULT_MODEL

_DESCRIPTION = (
    "Designs and updates personalised diet plans aligned with the user's goals, "
    "preferences, and any dietary restrictions. Can adjust meals on the fly and "
    "analyse food photos in future iterations."
)

_SYSTEM_PROMPT = """
You are the NutritionAgent for the GoGain virtual coach.

Responsibilities:
1. Create a macro- and calorie-aligned diet plan based on the training goal.
2. Allow substitutions while maintaining nutritional targets.
3. When the user shares a meal photo (future), analyse it and estimate macros.
4. Produce a daily nutrition recap and suggestions for improvement.
Output diet plans with a ```json diet_plan``` block for storage.
"""


class NutritionAgent:
    """LangChain implementation of the nutrition specialist agent."""

    agent_name = "NutritionAgent"

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        *,
        temperature: float = 0.6,
    ) -> None:
        self.logger = logging.getLogger(__name__)

        # All tools are now native LangChain StructuredTool instances
        self.tools = []

        self.llm = llm or ChatOpenAI(
            model=DEFAULT_MODEL,
            temperature=temperature,
            streaming=True,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                ("system", "{context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
        )

        self.logger.info(
            "🏗️ %s initialised with %d tools", self.agent_name, len(self.tools)
        )

    @staticmethod
    def _chunk_to_text(chunk: Any) -> str:
        if chunk is None:
            return ""

        content = getattr(chunk, "content", None)
        if not content:
            text_attr = getattr(chunk, "text", None)
            if text_attr:
                return text_attr
            return str(chunk)

        if isinstance(content, str):
            return content

        pieces: List[str] = []
        for part in content:
            if isinstance(part, dict):
                text_value = part.get("text")
                if text_value:
                    pieces.append(text_value)
            else:
                text_value = getattr(part, "text", None)
                if text_value:
                    pieces.append(text_value)
                elif isinstance(part, str):
                    pieces.append(part)
        return "".join(pieces)

    def _prepare_prompt_inputs(
        self,
        context_messages: Iterable[Dict[str, Any]],
        user_input: str,
    ) -> Dict[str, Any]:
        system_parts: List[str] = []
        conversation: List[Dict[str, Any]] = []

        for message in context_messages:
            if message.get("role") == "system":
                system_parts.append(message.get("content", ""))
            else:
                conversation.append(message)

        if (
            conversation
            and conversation[-1].get("role") == "user"
            and conversation[-1].get("content") == user_input
        ):
            conversation = conversation[:-1]

        chat_history: List[BaseMessage] = convert_dict_messages_to_langchain(conversation)
        context_text = "\n\n".join(part for part in system_parts if part).strip()

        return {
            "chat_history": chat_history,
            "context": context_text,
        }

    async def astream_response(
        self,
        context_messages: Iterable[Dict[str, Any]],
        user_input: str,
        **_: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        prompt_inputs = self._prepare_prompt_inputs(context_messages, user_input)
        inputs = {
            "input": user_input,
            **prompt_inputs,
        }

        buffer: List[str] = []

        async for event in self.executor.astream_events(inputs, version="v1"):
            kind = event.get("event")
            data = event.get("data") or {}

            if kind == "on_llm_stream":
                chunk = data.get("chunk")
                text = self._chunk_to_text(chunk)
                if text:
                    buffer.append(text)
                    yield {"type": "text", "content": text}

            elif kind == "on_tool_end":
                output = data.get("output")
                tool_name = event.get("name") or getattr(data.get("tool"), "name", None)
                if output:
                    text_output = output if isinstance(output, str) else str(output)
                    buffered = text_output.strip()
                    if buffered:
                        buffer.append(buffered)
                        yield {"type": "text", "content": buffered}

            elif kind == "on_llm_end":
                message = data.get("output")
                text = self._chunk_to_text(message)
                if text and not buffer:
                    buffer.append(text)
                    yield {"type": "text", "content": text}

            elif kind == "on_chain_end":
                output_payload = data.get("output")
                if isinstance(output_payload, dict):
                    final_text = (
                        output_payload.get("output")
                        or output_payload.get("final_output")
                        or ""
                    )
                else:
                    final_text = str(output_payload) if output_payload else ""

                if not final_text:
                    final_text = "".join(buffer)

                yield {
                    "type": "final",
                    "content": final_text,
                    "intermediate_steps": data.get("intermediate_steps"),
                }

    async def ainvoke(
        self,
        context_messages: Iterable[Dict[str, Any]],
        user_input: str,
        **_: Any,
    ) -> Dict[str, Any]:
        prompt_inputs = self._prepare_prompt_inputs(context_messages, user_input)
        result = await self.executor.ainvoke(
            {
                "input": user_input,
                **prompt_inputs,
            }
        )
        if not isinstance(result, dict):
            return {"output": str(result)}
        return result


def build(llm: Optional[Any] = None, **kwargs: Any) -> NutritionAgent:
    """Compatibility factory returning the LangChain nutrition agent."""
    if llm is not None and not isinstance(llm, ChatOpenAI):
        logging.getLogger(__name__).warning(
            "Ignoring non-LangChain llm passed to nutrition.build: %s", type(llm)
        )
        llm = None
    return NutritionAgent(llm=llm, **kwargs)

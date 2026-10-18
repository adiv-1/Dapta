from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .dapta_client import DaptaClient
from .orchestrator import LoraFactoryOrchestrator
from .session_store import SessionStore


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def build_orchestrator() -> LoraFactoryOrchestrator:
    required = (
        "DAPTA_AGENT_SEED",
        "DAPTA_CONTROLLER_URL",
        "DAPTA_CONTROL_TOKEN",
        "DAPTA_QWEN_URL",
        "DAPTA_QWEN_TOKEN",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError("Missing required configuration: " + ", ".join(missing))
    client = DaptaClient(
        controller_url=os.environ["DAPTA_CONTROLLER_URL"],
        controller_token=os.environ["DAPTA_CONTROL_TOKEN"],
        qwen_url=os.environ["DAPTA_QWEN_URL"],
        qwen_token=os.environ["DAPTA_QWEN_TOKEN"],
    )
    session_path = Path(os.environ.get("DAPTA_ASI1_STATE_PATH", "artifacts/asi1-agent.sqlite3"))
    session_path.parent.mkdir(parents=True, exist_ok=True)
    return LoraFactoryOrchestrator(
        client,
        SessionStore(session_path),
        default_adapter_id=os.environ.get("DAPTA_DEFAULT_ADAPTER_ID") or None,
        factory_builds_enabled=_enabled("DAPTA_ASI1_ENABLE_FACTORY_BUILDS"),
    )


def main() -> None:
    # Imported lazily so unit tests for the orchestration layer do not need the
    # Agentverse runtime installed in DAPTA's existing backend environment.
    from uagents import Agent, Context, Protocol
    from uagents_core.contrib.protocols.chat import (
        ChatAcknowledgement,
        ChatMessage,
        TextContent,
        chat_protocol_spec,
    )

    orchestrator = build_orchestrator()
    agent = Agent(
        name=os.environ.get("DAPTA_AGENT_NAME", "dapta-lora-factory"),
        seed=os.environ["DAPTA_AGENT_SEED"],
        port=int(os.environ.get("DAPTA_AGENT_PORT", "8003")),
        mailbox=True,
        publish_agent_details=True,
    )
    chat = Protocol(spec=chat_protocol_spec)

    @chat.on_message(ChatMessage)
    async def handle_message(ctx: Context, sender: str, message: ChatMessage) -> None:
        await ctx.send(
            sender,
            ChatAcknowledgement(
                timestamp=datetime.now(timezone.utc),
                acknowledged_msg_id=message.msg_id,
            ),
        )
        text = message.text() if hasattr(message, "text") else " ".join(
            item.text for item in message.content if isinstance(item, TextContent)
        )
        replies = await asyncio.to_thread(orchestrator.handle, sender, text)
        for reply in replies:
            await ctx.send(
                sender,
                ChatMessage(
                    timestamp=datetime.now(timezone.utc),
                    msg_id=uuid4(),
                    content=[TextContent(type="text", text=reply)],
                ),
            )

    @chat.on_message(ChatAcknowledgement)
    async def handle_acknowledgement(
        ctx: Context, sender: str, message: ChatAcknowledgement
    ) -> None:
        del ctx, sender, message

    agent.include(chat, publish_manifest=True)
    agent.run()


if __name__ == "__main__":
    main()

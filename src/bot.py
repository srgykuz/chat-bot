import logging
from typing import Dict, Any, Optional

from src.llm import ModelClient
from src.session import Message, MessageRole, SessionClient, Persona, User
from src.telegram import TelegramClient, TelegramMessage, parse_update


logger = logging.getLogger(__name__)

telegram_client = TelegramClient()
session_client = SessionClient()
model_client = ModelClient()


async def aclose() -> None:
    """
    Closes used clients.
    """
    await telegram_client.aclose()
    session_client.close()
    model_client.close()


async def handle_update(update: Dict[str, Any]) -> None:
    """
    Handles Telegram update item.
    """
    message = parse_update(update)

    if (not message) or (message.chat_id is None) or (message.text is None):
        logger.info(f"Unsupported update: {update}")
        return

    logger.info(f"Processing update {message.update_id} from {message.username}: {message.text}")

    if message.text.startswith("/"):
        await handle_command(message)
    else:
        await handle_message(message)


async def handle_command(message: TelegramMessage) -> None:
    """
    Handles a message that contain app command text.
    """
    chat_id = message.chat_id or 0
    text = (message.text or "").strip()

    command = text.split()[0].lower()
    response = ""

    if command == "/get_persona":
        persona = session_client.get_persona(chat_id)

        if persona:
            response = (
                "*Current persona:*\n"
                f"Name: `{persona.name}`\n"
            )
        else:
            response = "No persona is currently selected for this chat."
    elif command == "/clear_persona":
        session_client.clear(chat_id)
        response = "Persona cleared. A new persona will be created on the next message."
    elif command == "/set_persona":
        parts = text.split(maxsplit=1)

        if len(parts) < 2 or not parts[1].strip():
            response = "Usage: /set\\_persona <Name>"
        else:
            name = parts[1].strip()
            persona: Optional[Persona] = None

            try:
                persona = session_client.select_persona(name)
            except Exception:
                persona = None

            if persona:
                session_client.set_persona(chat_id, persona)
                response = f"Persona set to {persona.name}."
            else:
                response = f"Persona {name} not found."
    elif command == "/list_persona":
        names = [p.name for p in session_client.load_personas()]

        if names:
            names = [f"`{n}`" for n in names]
            response = "*Available personas:*\n" + "\n".join(names)
        else:
            response = "No personas available."
    elif command == "/get_history":
        info = session_client.get_history_info(chat_id)
        response = (
            "*Chat history info:*\n"
            f"Total messages: `{info.num_messages}`\n"
            f"Max history stored: `{info.max_messages}`\n"
            f"User messages: `{info.num_user_messages}`\n"
            f"Assistant messages: `{info.num_assistant_messages}`"
        )
    elif command == "/clear_history":
        session_client.clear(chat_id)
        response = "Chat history cleared."
    elif command == "/clear":
        session_client.clear(chat_id)
        response = "Session cleared."
    else:
        response = (
            "Persona commands:\n"
            "/set\\_persona <name>\n"
            "/get\\_persona\n"
            "/list\\_persona\n"
            "/clear\\_persona\n"
            "\n"
            "History commands:\n"
            "/get\\_history\n"
            "/clear\\_history\n"
            "\n"
            "Other commands:\n"
            "/clear"
        )

    await telegram_client.send_message(
        chat_id=chat_id,
        text=response,
        reply_to_message_id=message.message_id,
        escape=False
    )


async def handle_message(message: TelegramMessage) -> None:
    """
    Handles a message that contain plain text a LLM should respond to in the chat context.
    """
    chat_id = message.chat_id or 0
    text = (message.text or "").strip()

    await telegram_client.send_chat_action(chat_id, action="typing")

    history = session_client.get_history(chat_id)
    history.append(Message(role=MessageRole.USER, content=text))

    persona = session_client.init_persona(chat_id)
    user = User(
        first_name=message.first_name,
        last_name=message.last_name,
    )
    system_prompt = model_client.build_system_prompt(persona, user)

    response = ""
    success = False

    try:
        response = await model_client.chat(system_prompt, history)
        success = True
    except Exception as e:
        response = "🤖"
        success = False
        logger.error("LLM call error: %s", e, exc_info=True)

    if success:
        session_client.append_history(chat_id, Message(role=MessageRole.USER, content=text))
        session_client.append_history(chat_id, Message(role=MessageRole.ASSISTANT, content=response))

    await telegram_client.send_message(chat_id=chat_id, text=response)
    logger.info(f"Responding to update {message.update_id} from {message.username}: {response}")

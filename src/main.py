import asyncio
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from src.config import configure_logger, get_logger, get_settings, get_redis
from src.bot import handle_update, aclose as bot_aclose
from src.telegram import TelegramPoller
from src.session import validate_personas
from src.analytics import close as analytics_close
from src.proactivity import close as proactivity_close


configure_logger()

logger = get_logger(__name__)
app = FastAPI()
settings = get_settings()

app.state.poller = None
app.state.poller_task = None


@app.on_event("startup")
async def on_startup():
    validate_personas()

    if not settings.telegram_webhook_enable:
        logger.info("Starting Telegram long polling")
        app.state.poller = TelegramPoller(handle_update)
        app.state.poller_task = asyncio.create_task(app.state.poller.start())


@app.on_event("shutdown")
async def on_shutdown():
    if app.state.poller_task:
        app.state.poller_task.cancel()

        try:
            await app.state.poller_task
        except asyncio.CancelledError:
            pass

    if app.state.poller:
        await app.state.poller.aclose()

    await bot_aclose()
    analytics_close()
    proactivity_close()
    get_redis().close()


@app.get("/", status_code=404, response_class=PlainTextResponse)
async def root() -> str:
    return "not found"


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True}


@app.post("/webhook")
async def webhook(request: Request) -> JSONResponse:
    if settings.telegram_webhook_secret_token:
        secret_token = request.headers.get("X-Telegram-Bot-Api-Secret-Token")

        if secret_token != settings.telegram_webhook_secret_token:
            return JSONResponse({"ok": False}, status_code=401)

    update = await request.json()

    try:
        await handle_update(update)
    except Exception as e:
        logger.error(f"Error handling webhook update: {e}", exc_info=True)

    return JSONResponse({"ok": True})

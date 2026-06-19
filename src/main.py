import asyncio
import logging
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from src.config import get_settings, get_redis, get_httpx
from src.bot import handle_update, aclose as bot_aclose
from src.telegram import TelegramPoller
from src.analytics import close as analytics_close


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
app = FastAPI(title="Chat Bot")
settings = get_settings()

app.state.poller = None
app.state.poller_task = None


@app.on_event("startup")
async def on_startup():
    if settings.telegram_use_polling:
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
    await get_httpx().aclose()
    get_redis().close()


@app.get("/", status_code=404, response_class=PlainTextResponse)
async def root() -> str:
    return "not found"


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True}


@app.post("/webhook")
async def webhook(request: Request) -> JSONResponse:
    update = await request.json()
    await handle_update(update)

    return JSONResponse({"ok": True})

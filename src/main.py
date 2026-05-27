"""Main FastAPI application for Friend Bot."""
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import asyncio
import logging
from typing import Dict, Any
from src.config import get_settings
from src.bot import process_update
from src.poller import TelegramPoller

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Friend Bot", version="0.1.0")

settings = get_settings()
app.state.poller_task = None


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("Friend Bot starting up...")

    if settings.telegram_use_polling:
        logger.info("Starting Telegram long polling in development mode")
        app.state.poller_task = asyncio.create_task(TelegramPoller().start())


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Friend Bot shutting down...")
    if app.state.poller_task:
        app.state.poller_task.cancel()
        try:
            await app.state.poller_task
        except asyncio.CancelledError:
            pass


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


@app.post("/webhook")
async def webhook(request: Request) -> JSONResponse:
    """Receive and process Telegram updates via webhook.

    This endpoint receives updates from Telegram when a user sends a message.
    """
    try:
        update = await request.json()
        logger.info(f"Received update: {update.get('update_id')}")

        await process_update(update)
        return JSONResponse({"ok": True})

    except Exception as e:
        logger.error(f"Error processing webhook: {e}", exc_info=True)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {
        "message": "Friend Bot API",
        "version": "0.1.0",
        "status": "running"
    }


@app.post("/set-webhook")
async def set_webhook_endpoint(webhook_url: str) -> Dict[str, Any]:
    """Manually set the webhook URL.

    This is useful for initial setup or testing.

    Args:
        webhook_url: Full URL where Telegram should send updates
    """
    try:
        from src.telegram_handler import TelegramHandler

        result = await TelegramHandler().set_webhook(webhook_url)
        logger.info(f"Webhook set to {webhook_url}")
        return result
    except Exception as e:
        logger.error(f"Error setting webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/webhook-info")
async def webhook_info() -> Dict[str, Any]:
    """Get current webhook information."""
    try:
        from src.telegram_handler import TelegramHandler

        info = await TelegramHandler().get_webhook_info()
        return info
    except Exception as e:
        logger.error(f"Error getting webhook info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

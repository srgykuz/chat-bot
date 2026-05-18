# Friend Bot 🤖

A Telegram bot that acts as a virtual friend with its own identity and personality. Have conversations just like you would with a real friend.

## Features

- ✅ Telegram webhook integration for real-time message handling
- ✅ Per-user conversation state and persona management
- ✅ Integration with Gemini API and OpenAI API
- ✅ Redis for caching and session storage
- ✅ Background job queue (RQ) for async tasks and scheduled messages
- ✅ Docker Compose deployment
- ✅ Unique personality for each user

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- Telegram Bot Token (from BotFather)
- LLM API key (Gemini or OpenAI)

### Setup

1. **Clone and setup the project:**
   ```bash
   cd friend-bot
   cp .env.example .env
   ```

2. **Configure environment variables in `.env`:**
   ```bash
   # Get your bot token from Telegram BotFather
   TELEGRAM_TOKEN=your_bot_token_here
   TELEGRAM_WEBHOOK_URL=https://yourdomain.com/webhook

   # Choose your LLM provider
   GEMINI_API_KEY=your_key_here
   # OR
   OPENAI_API_KEY=your_key_here
   ```

3. **Start with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

   This will start:
   - Redis instance on port 6379
   - FastAPI server on port 8000
   - Background worker for async tasks

4. **Initialize webhook (on production with public URL):**
   ```bash
   curl -X POST "http://localhost:8000/set-webhook?webhook_url=https://yourdomain.com/webhook"
   ```

### Local Development (with polling)

For prototyping without a public webhook URL:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Redis locally or use Docker
docker run -d -p 6379:6379 redis:7-alpine

# Create .env file with your credentials
cp .env.example .env
# Edit .env with your tokens

# Run the API server
uvicorn src.main:app --reload

# In another terminal, run the worker
python -m src.worker
```

### Testing the Bot

1. **Health check:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Send a test message (local polling simulation):**
   ```bash
   curl -X POST http://localhost:8000/webhook \
     -H "Content-Type: application/json" \
     -d '{
       "update_id": 123,
       "message": {
         "message_id": 1,
         "date": 1234567890,
         "chat": {"id": 12345},
         "from": {
           "id": 67890,
           "first_name": "John",
           "username": "john_doe"
         },
         "text": "Hello friend!"
       }
     }'
   ```

3. **Check webhook status:**
   ```bash
   curl http://localhost:8000/webhook-info
   ```

## Project Structure

```
friend-bot/
├── src/
│   ├── __init__.py           # Package init
│   ├── main.py              # FastAPI app and webhook handler
│   ├── config.py            # Configuration management
│   ├── telegram_handler.py  # Telegram API wrapper
│   └── worker.py            # Background task worker
├── docker-compose.yml       # Docker services orchestration
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
├── .env.example           # Environment template
├── .gitignore            # Git ignore rules
└── README.md            # This file
```

## Architecture Overview

### Current State (Base)
1. **Telegram Webhook** - Receives messages from Telegram
2. **Echo Handler** - Currently replies with echoed messages
3. **Redis Cache** - Available for session storage
4. **RQ Worker** - Ready for background tasks

### Next Steps
- Integrate LLM API (Gemini/OpenAI)
- Implement per-user session management
- Add persona system and context management
- Build conversation memory with summaries
- Add proactive message scheduling
- Multi-user isolation and uniqueness

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| POST | `/webhook` | Telegram updates receiver |
| POST | `/set-webhook` | Manually set webhook URL |
| GET | `/webhook-info` | Get webhook status |

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `TELEGRAM_TOKEN` | Telegram bot token from BotFather | Yes |
| `TELEGRAM_WEBHOOK_URL` | Webhook URL for production | Yes |
| `GEMINI_API_KEY` | Google Gemini API key | No |
| `OPENAI_API_KEY` | OpenAI API key | No |
| `REDIS_URL` | Redis connection URL | No (default: localhost:6379) |
| `ENVIRONMENT` | Environment mode (development/production) | No |

## Deployment

### Using Docker Compose (Production)

1. Configure environment variables in `.env`
2. Ensure your domain has a valid SSL certificate
3. Update TELEGRAM_WEBHOOK_URL with your public domain
4. Run:
   ```bash
   docker-compose up -d
   ```

### Setting Webhook on Production

After deploying, set the webhook with:
```bash
curl -X POST "https://yourdomain.com/set-webhook?webhook_url=https://yourdomain.com/webhook"
```

## Troubleshooting

### Webhook not receiving messages
- Check webhook status: `curl http://localhost:8000/webhook-info`
- Ensure your domain is publicly accessible
- Verify SSL certificate is valid
- Check Telegram firewall whitelist

### Redis connection issues
- Ensure Redis is running: `docker-compose ps`
- Check REDIS_URL is correct
- Verify Redis port (default 6379) is not blocked

### API not responding
- Check logs: `docker-compose logs api`
- Ensure port 8000 is not in use
- Verify all environment variables are set

## Best Practices

- 🔒 Never commit `.env` file with real tokens
- 📝 Keep logs for debugging and monitoring
- 🚀 Use environment-specific configurations
- 💾 Regularly backup Redis data
- 🔄 Keep conversation history clean and concise
- 🧬 Preserve bot personality across sessions

## Next Implementation Phases

1. **Phase 2**: LLM Integration & Context Management
   - Integrate Gemini/OpenAI API
   - Implement user sessions in Redis
   - Build conversation history storage

2. **Phase 3**: Persona & Memory System
   - Generate unique personas per user
   - Store and retrieve user profiles
   - Implement long-term memory with summaries

3. **Phase 4**: Proactive Messaging
   - Schedule background messages
   - Implement cooldown and timezone support
   - Add user preferences

4. **Phase 5**: Advanced Features
   - Multi-user scaling
   - Analytics and monitoring
   - Commands system (/reset, /history, etc.)

## License

MIT

## Contributing

Feel free to submit issues and enhancement requests!

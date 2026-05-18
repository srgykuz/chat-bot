# Getting Started with Friend Bot

## Step 1: Get Your Telegram Bot Token

1. Open Telegram and search for `@BotFather`
2. Click `/start` and follow the prompts
3. Send `/newbot` to create a new bot
4. Follow the setup wizard (give it a name and username)
5. Copy the token that BotFather gives you (looks like: `123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11`)

## Step 2: Get an LLM API Key

Choose at least one:

### Option A: Google Gemini
1. Go to https://console.cloud.google.com/
2. Create a new project or select existing
3. Enable the Generative AI API
4. Create an API key
5. Copy the key

### Option B: OpenAI
1. Go to https://platform.openai.com/
2. Create an account or login
3. Navigate to API keys section
4. Create a new secret key
5. Copy the key

## Step 3: Setup the Project

### Using Docker Compose (Recommended)

```bash
# 1. Clone the repository
cd friend-bot

# 2. Create .env file with your credentials
cp .env.example .env

# 3. Edit .env and add your tokens
# - TELEGRAM_TOKEN: your token from BotFather
# - GEMINI_API_KEY or OPENAI_API_KEY: your LLM key
nano .env  # or use your editor

# 4. Start the services
docker-compose up -d

# 5. Verify everything is running
docker-compose ps
```

You should see:
- `friend-bot-redis` - running on port 6379
- `friend-bot-api` - running on port 8000
- `friend-bot-worker` - running in background

### For Local Development (Without Docker)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Redis (in one terminal)
docker run -d -p 6379:6379 redis:7-alpine

# 4. Create .env with your credentials
cp .env.example .env
# Edit .env - keep REDIS_URL as default for local Redis

# 5. Start API server (in one terminal)
uvicorn src.main:app --reload

# 6. Start worker (in another terminal)
python -m src.worker
```

## Step 4: Test the Setup

### Quick Health Check
```bash
curl http://localhost:8000/health
# Should return: {"status":"ok","version":"0.1.0"}
```

### Send a Test Message
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
        "first_name": "Test",
        "username": "testuser"
      },
      "text": "Hello!"
    }
  }'
```

## Step 5: Set Up Webhook (Production Only)

For local development, the bot uses polling simulation. For production:

1. Get a public domain (e.g., `yourdomain.com`)
2. Ensure it has a valid SSL certificate
3. Set the webhook:
   ```bash
   curl -X POST "https://yourdomain.com/set-webhook?webhook_url=https://yourdomain.com/webhook"
   ```
4. Verify:
   ```bash
   curl https://yourdomain.com/webhook-info
   ```

## Step 6: Test with Telegram

1. Find your bot: search for your username in Telegram
2. Send it a message
3. You should get an echo response

## Using Make Commands

If you have `make` installed:

```bash
make setup      # Create .env
make up         # Start services
make down       # Stop services
make logs       # View all logs
make api-logs   # View API logs only
make clean      # Clean up containers
```

## Troubleshooting

### "Connection refused" error
- Ensure Docker is running: `docker --version`
- Check if port 8000 is available: `lsof -i :8000`
- Check Redis: `docker-compose logs redis`

### "TELEGRAM_TOKEN" error
- Verify .env file exists and has correct format
- Check the token is correct from BotFather
- No quotes needed around the token

### No response from bot
- Check webhook status: `curl http://localhost:8000/webhook-info`
- For local dev, you need to manually send test requests
- Check logs: `docker-compose logs api`

### Redis connection error
- Ensure Redis service is running: `docker-compose ps`
- Default local Redis URL is `redis://localhost:6379`
- Check REDIS_URL in .env

## Next Steps

Once the base is working:

1. **Phase 2**: Implement LLM integration
   - Replace echo with actual API calls
   - Store user conversations
   - Build context management

2. **Phase 3**: Add persona system
   - Generate unique friend identities
   - Store per-user personality data
   - Implement memory and context

3. **Phase 4**: Proactive messaging
   - Add scheduled messages
   - Implement cooldown periods
   - Add timezone support

## Project Structure

```
friend-bot/
├── src/
│   ├── main.py              # FastAPI app and endpoints
│   ├── config.py            # Settings management
│   ├── telegram_handler.py  # Telegram API integration
│   └── worker.py            # Background tasks
├── docker-compose.yml       # Service orchestration
├── requirements.txt         # Python packages
├── .env                     # Your credentials (don't commit!)
└── README.md               # Full documentation
```

## Getting Help

- Check logs: `docker-compose logs -f`
- Read the main [README.md](README.md)
- Review API endpoints in [README.md](README.md#api-endpoints)

## Important Security Notes

⚠️ **Never commit `.env` file with real tokens!**
- `.gitignore` already excludes it
- Keep your API keys private
- Use environment variables in production
- Rotate tokens if they're ever exposed

Happy bot building! 🤖

"""Background worker for async tasks using RQ."""
import logging
from redis import Redis
from rq import Worker, Queue
from src.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


def start_worker():
    """Start the RQ worker."""
    redis_url = settings.redis_url
    redis_conn = Redis.from_url(redis_url, decode_responses=True)

    # Create queues for different task types
    queues = [
        Queue("default", connection=redis_conn),
        Queue("messages", connection=redis_conn),
        Queue("scheduled", connection=redis_conn),
    ]

    worker = Worker(queues, connection=redis_conn)
    logger.info("Starting RQ worker...")
    worker.work()


if __name__ == "__main__":
    start_worker()

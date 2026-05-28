import logging

from redis import Redis
from rq import Worker, Queue

from src.config import get_settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
settings = get_settings()


def main():
    """
    Starts RQ worker to process background tasks.
    """
    redis_conn = Redis.from_url(settings.redis_url, decode_responses=True)
    queues = [
        Queue("default", connection=redis_conn),
    ]

    logger.info("Starting RQ worker for queues: %s", [q.name for q in queues])

    worker = Worker(queues, connection=redis_conn)
    worker.work()


if __name__ == "__main__":
    main()

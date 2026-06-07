import logging

from rq import Worker

from src.config import get_redis, get_queue


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Starts RQ worker and scheduler.
    """
    redis = get_redis(decode_responses=False)
    queue = get_queue()

    logger.info(f"Starting RQ worker for queue: {queue.name}")

    worker = Worker(queue, connection=redis)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()

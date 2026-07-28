from rq import Worker

from src.config import configure_logger, get_logger, get_redis, get_queue


configure_logger()

logger = get_logger(__name__)


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

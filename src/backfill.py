import argparse
import asyncio
import logging
from datetime import datetime, timedelta, timezone

from . import ingest

logger = logging.getLogger(__name__)


def backfill(start_date: datetime, end_date: datetime | None = None) -> None:
    """Backfill btc_weekly data from ``start_date`` up to ``end_date``."""
    end_date = end_date or datetime.now(timezone.utc)
    anchor = (start_date - timedelta(days=start_date.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    end_limit = (end_date - timedelta(days=end_date.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    while anchor <= end_limit:
        logger.info("Ingesting week starting %s", anchor.date())
        asyncio.run(ingest.ingest_weekly(week_anchor=anchor))
        asyncio.run(asyncio.sleep(0.5))
        anchor += timedelta(days=7)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill btc_weekly history")
    parser.add_argument(
        "--years", type=int, default=15, help="Number of years to backfill"
    )
    args = parser.parse_args()

    start = datetime.now(timezone.utc) - timedelta(days=args.years * 365)
    backfill(start)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys

if __package__ in {None, ""}:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from whale_tracker.alerter import TelegramAlerter
from whale_tracker.config import Config
from whale_tracker.db import Database
from whale_tracker.monitor import WhaleMonitor
from whale_tracker.scanner import TraderScanner


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("whale_tracker.main")


async def run(args: argparse.Namespace) -> None:
    config = Config.from_env()
    if args.telegram_token:
        config.telegram_token = args.telegram_token
    if args.channel_id:
        config.channel_id = args.channel_id
    if args.db_path:
        config.db_path = args.db_path

    db = Database(config.db_path)
    scanner = TraderScanner(db, config)
    alerter = TelegramAlerter(
        db=db,
        config=config,
        token=config.telegram_token,
        channel_id=config.channel_id,
        dry_run=args.dry_run,
    )
    monitor = WhaleMonitor(db, alerter, config)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            pass

    try:
        if args.list_traders:
            rows = db.list_all_traders(limit=500)
            for row in rows:
                flag = "*" if row["is_monitored"] else " "
                print(
                    f"{flag} {row['address'][:12]}... "
                    f"score={float(row['score'] or 0):.1f} "
                    f"wr={float(row['win_rate'] or 0)*100:.1f}% "
                    f"30d={float(row['pnl_30d'] or 0):.0f}"
                )
            return

        if args.stats:
            stats = db.stats()
            print("Whale tracker stats")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            return

        if args.add_trader:
            profile = await scanner.analyze_trader(args.add_trader)
            db.upsert_trader(profile)
            db.set_monitored(profile.address, True)
            print(
                f"Added {profile.address[:12]}... | score={profile.score:.1f} "
                f"wr={profile.win_rate*100:.1f}% pnl30d={profile.pnl_30d:.0f}"
            )
            return

        monitored_count = db.count_monitored_traders()

        if monitored_count == 0:
            log.info("No monitored traders in DB. Running initial scan.")
            await scanner.full_scan()
            monitored_count = db.count_monitored_traders()
        else:
            log.info("Found %d monitored traders in DB. Skipping startup scan.", monitored_count)

        if args.scan_only:
            log.info("Scan complete. Exiting scan-only mode.")
            return

        log.info(
            "Starting whale tracker | mode=%s monitored=%d poll=%.1fs",
            "DRY-RUN" if args.dry_run else "LIVE",
            monitored_count,
            config.poll_interval_seconds,
        )

        # Refresh once per 24h (not on every restart)
        monitor_task = asyncio.create_task(monitor.poll_loop())
        scanner_task = asyncio.create_task(scanner.periodic_refresh(24))
        stop_task = asyncio.create_task(stop_event.wait())

        done, pending = await asyncio.wait(
            {monitor_task, scanner_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if stop_task in done:
            log.info("Stop signal received. Shutting down.")

        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    finally:
        await alerter.flush_due(force=True)
        await alerter.close()
        await monitor.close()
        await scanner.close()
        db.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HyperPulse whale tracker")
    parser.add_argument("--scan-only", action="store_true", help="Run scanner once and exit")
    parser.add_argument("--dry-run", action="store_true", help="Do not send Telegram alerts")
    parser.add_argument("--add-trader", type=str, help="Add one trader to monitored list")
    parser.add_argument("--list-traders", action="store_true", help="List traders in DB")
    parser.add_argument("--stats", action="store_true", help="Show database stats")
    parser.add_argument("--telegram-token", type=str, default="", help="Override bot token")
    parser.add_argument("--channel-id", type=str, default="", help="Override channel id")
    parser.add_argument("--db-path", type=str, default="", help="Override SQLite db path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

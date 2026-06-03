from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from ingestion.api_client import run_ingestion
from ingestion.normalizer import normalize_all
from ingestion.graph_writer import write_all_trials
from ingestion.vector_writer import write_all_to_qdrant
from datetime import datetime
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONDITIONS = ["cancer", "diabetes", "alzheimer"]
MAX_PER_CONDITION = 50


def run_pipeline():
    print(f"\n{'='*50}")
    print(f"Pipeline started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")

    try:
        # Step 1 — fetch
        print("\n[1/3] Fetching trials...")
        raw = run_ingestion(CONDITIONS, max_per_condition=MAX_PER_CONDITION)
        if not raw:
            print("No trials fetched — skipping rest of pipeline")
            return

        # Step 2 — normalize
        print("\n[2/3] Normalizing...")
        records = normalize_all(raw)
        if not records:
            print("No valid records after normalization — skipping")
            return

        # Step 3 — write to both stores
        print("\n[3/3] Writing to Neo4j and Qdrant...")
        write_all_trials(records)
        write_all_to_qdrant(records)

        print(f"\nPipeline complete — {len(records)} trials processed")
        print(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)


def start_scheduler():
    scheduler = BlockingScheduler()
    scheduler.add_job(
        run_pipeline,
        trigger="interval",
        hours=6,
        next_run_time=datetime.now(),  # run immediately on start
        id="ingestion_pipeline",
    )
    print("Scheduler started — pipeline runs every 6 hours")
    print("Press Ctrl+C to stop\n")
    try:
        scheduler.start()
    except KeyboardInterrupt:
        print("\nScheduler stopped")


def get_background_scheduler() -> BackgroundScheduler:
    """Returns a background scheduler for use inside FastAPI."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        run_pipeline,
        trigger="interval",
        hours=6,
        id="ingestion_pipeline",
    )
    return scheduler


if __name__ == "__main__":
    run_pipeline()
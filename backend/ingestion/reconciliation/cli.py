import argparse
import asyncio
import json
import logging
import sys
from backend.ingestion.reconciliation.reconciler import IngestionReconciler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reconciliation-cli")


async def main_async(audit_only: bool, repair: bool) -> None:
    reconciler = IngestionReconciler()
    logger.info("Starting ingestion health audit...")
    issues = await reconciler.run_audit()

    print("\n--- INGESTION HEALTH AUDIT REPORT ---")
    print(json.dumps(issues, indent=2))
    print("------------------------------------\n")

    total_issues = sum(len(v) for v in issues.values())
    if total_issues == 0:
        logger.info("All health checks PASSED. No anomalies detected.")
        return

    logger.warning(f"Detected {total_issues} anomalies across the ingestion manifest.")

    if repair:
        logger.info("Applying automated repairs...")
        repaired_summary = await reconciler.run_repair(issues)
        print("\n--- REPAIR ACTIONS APPLIED ---")
        print(json.dumps(repaired_summary, indent=2))
        print("------------------------------\n")
        logger.info("Repairs completed successfully.")
    else:
        logger.info("Run with --repair to automatically fix recoverable anomalies.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial RAG Ingestion Health & Reconciliation CLI")
    parser.add_argument("--audit", action="store_true", help="Run read-only audit")
    parser.add_argument("--repair", action="store_true", help="Run audit and apply repairs")

    args = parser.parse_args()
    if not args.audit and not args.repair:
        parser.print_help()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main_async(audit_only=args.audit, repair=args.repair))


if __name__ == "__main__":
    main()

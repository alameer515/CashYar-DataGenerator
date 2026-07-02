"""
main.py
--------------------------------

Run the complete CashYar synthetic dataset pipeline.
"""

import time

from generate_behavior_summary import BehaviorSummaryGenerator
from generate_transactions import TransactionGenerator
from generate_users import UserGenerator
from utils.helpers import ensure_data_dir, print_banner


def main():
    start = time.time()
    ensure_data_dir("data")

    print_banner("CashYar Synthetic Dataset Generator")

    print("Step 1/3: Generating users...")
    users = UserGenerator().generate_dataset()
    users.to_csv("data/users.csv", index=False)
    print(f"  -> {len(users):,} users saved to data/users.csv")

    print()
    print("Step 2/3: Generating transactions...")
    tx_generator = TransactionGenerator()
    transactions = tx_generator.generate_dataset()
    tx_generator.save(transactions)

    print()
    print("Step 3/3: Generating behavioral summaries...")
    summary_generator = BehaviorSummaryGenerator()
    summaries = summary_generator.generate_dataset()
    summary_generator.save(summaries)

    elapsed = round(time.time() - start, 1)
    print()
    print_banner(f"Dataset generation complete in {elapsed}s")
    print("Output files:")
    print("  - data/users.csv")
    print("  - data/transactions.csv")
    print("  - data/behavioral_summary.csv")


if __name__ == "__main__":
    main()

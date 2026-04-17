#!/usr/bin/env python3
"""
Developer Helper Script for STOCK MARKET Pro
Resets all freemium state for testing.
Passcode: 4594
"""

import os
import sys
import shutil
from pathlib import Path

PASSCODE = "4594"

def main():
    print("🔐 Enter passcode to reset app state:")
    entered = input("Passcode: ").strip()
    if entered != PASSCODE:
        print("❌ Incorrect passcode. Exiting.")
        sys.exit(1)

    print("\n✅ Passcode accepted. Resetting state...\n")

    paths_to_delete = [
        "data/pro_us_market.db",
        "data/users.csv",
        "data/feedback.csv",
        "data/app.log",
        "data/cache",
        "data/exports",
        "data/reports"
    ]

    for p in paths_to_delete:
        path = Path(p)
        if path.exists():
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                    print(f"🗑️  Removed directory: {p}")
                else:
                    path.unlink()
                    print(f"🗑️  Removed file: {p}")
            except Exception as e:
                print(f"⚠️  Could not remove {p}: {e}")
        else:
            print(f"ℹ️  Not found (skip): {p}")

    for d in ["data/cache", "data/exports", "data/reports"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("📁 Recreated required directories.")

    streamlit_cache = Path.home() / ".streamlit" / "cache"
    if streamlit_cache.exists():
        try:
            shutil.rmtree(streamlit_cache)
            print(f"🧹 Cleared Streamlit cache: {streamlit_cache}")
        except Exception as e:
            print(f"⚠️  Could not clear Streamlit cache: {e}")

    print("\n🎉 Reset complete. You can now run the app with a fresh state.")
    print("   Run: streamlit run STOCK_MARKET_Pro.py")

if __name__ == "__main__":
    main()
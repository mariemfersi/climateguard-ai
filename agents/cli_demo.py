"""
Quick interactive demo of the Phase 9 multi-agent layer.

Usage (from repo root, with .env populated):
    python -m agents.cli_demo
"""

from __future__ import annotations

import logging

from agents.orchestrator.orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    print("ClimateGuard AI — Phase 9 Agent Demo")
    print("Type a question (or 'quit'). Examples:")
    print("  • What is the technical rate for a Florida wind layer attaching at 50m?")
    print("  • Does this treaty structure trigger Article 105 SCR?")
    print("  • What if Category 5 landfall frequency rises 20%?")
    print("  • Draft a full pricing memo for the current book.")
    print("-" * 60)

    orch = Orchestrator()

    while True:
        try:
            user = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user or user.lower() in {"q", "quit", "exit"}:
            print("Bye.")
            break

        resp = orch.route(user)
        print("\nAgent>")
        print(resp.text)
        print("\n--- meta ---")
        print(f"Fidelity passed : {resp.fidelity_passed}")
        print(f"Fidelity msg    : {resp.fidelity_message}")
        if resp.citations:
            print("Citations      :")
            for c in resp.citations:
                print(f"  • {c}")


if __name__ == "__main__":
    main()
"""Minimal test runner so the suite works without pytest installed.

    python3.12 -m tests.run          # standalone
    python3.12 -m pytest tests/      # also works if pytest is installed
"""

import importlib
import traceback

MODULES = ["tests.test_hdc", "tests.test_maxent", "tests.test_federated"]


def main():
    total = failed = 0
    for name in MODULES:
        mod = importlib.import_module(name)
        for attr in sorted(dir(mod)):
            if attr.startswith("test_"):
                total += 1
                try:
                    getattr(mod, attr)()
                    print(f"PASS  {name}.{attr}")
                except Exception:
                    failed += 1
                    print(f"FAIL  {name}.{attr}")
                    traceback.print_exc()
    print(f"\n{total - failed}/{total} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

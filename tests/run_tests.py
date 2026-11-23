#!/usr/bin/env python3
"""
Run the hi_cube_qa validation tests without needing pytest.

    python3 tests/run_tests.py            # all test_*.py files
    python3 tests/run_tests.py noise      # only files whose name contains 'noise'

(`python3 -m pytest tests/` also works wherever pytest is installed.)
"""

import glob
import importlib.util
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))


def run_module(mod) -> int:
    fails = 0
    for name in sorted(n for n in dir(mod) if n.startswith("test_")):
        fn = getattr(mod, name)
        if not callable(fn):
            continue
        t0 = time.time()
        try:
            fn()
            print(f"  PASS  {mod.__name__}::{name}  ({time.time() - t0:.1f}s)")
        except Exception:
            fails += 1
            print(f"  FAIL  {mod.__name__}::{name}")
            traceback.print_exc()
    return fails


def main(argv) -> int:
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
    sys.path.insert(0, HERE)
    pattern = argv[1] if len(argv) > 1 else ""
    total_fail = n_files = 0
    for path in sorted(glob.glob(os.path.join(HERE, "test_*.py"))):
        if pattern not in os.path.basename(path):
            continue
        spec = importlib.util.spec_from_file_location(
            os.path.splitext(os.path.basename(path))[0], path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        n_files += 1
        total_fail += run_module(mod)
    print(f"\n{n_files} file(s), {total_fail} failure(s)")
    return 1 if total_fail else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

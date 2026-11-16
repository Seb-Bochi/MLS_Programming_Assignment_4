"""Capture a child Python process without PowerShell treating warnings as errors."""
import os
import subprocess
import sys

os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
with open(sys.argv[1], "w", encoding="utf-8") as output:
    result = subprocess.run([sys.executable, "-u", *sys.argv[2:]],
                            stdout=output, stderr=subprocess.STDOUT)
print(f"Python exit code: {result.returncode}")
raise SystemExit(result.returncode)

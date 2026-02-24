"""Launcher that starts the bot as a detached subprocess."""
import subprocess
import sys
from pathlib import Path

bot_dir = Path(__file__).parent
proc = subprocess.Popen(
    [sys.executable, "run.py", "--no-dashboard"],
    cwd=str(bot_dir),
    stdin=subprocess.PIPE,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
)
proc.stdin.write(b"YES\n")
proc.stdin.close()
print(f"Bot launched as PID {proc.pid}")

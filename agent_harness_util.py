"""
Shared helpers for stopping API agent child processes between matches.

Used by agent_test.py and run.py so fixed ports (e.g. 8000/8001) are released
reliably before the next subprocess bind (avoids errno 98 cascades).
"""

import multiprocessing
import time
from typing import Optional


def shutdown_agent_process(proc: Optional[multiprocessing.Process], join_timeout: float = 12.0) -> None:
    if proc is None:
        return
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=join_timeout)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=4.0)
    else:
        proc.join(timeout=0.2)


def pause_for_port_release(seconds: float = 0.75) -> None:
    time.sleep(seconds)

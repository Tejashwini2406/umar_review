"""Production Operational CLI Prediction Entrypoint Wrapper.

Delegates execution directly to iot_ids.cli.
"""
from __future__ import annotations

import sys
from iot_ids.cli import main

if __name__ == "__main__":
    main()

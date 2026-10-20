"""Retired legacy paper-evidence compiler.

The old compiler embedded withdrawn metrics and labeled its output authoritative.
Rebuild a paper evidence package only after the remaining experiments are complete
and the input artifacts have been audited against the current protocol.
"""
from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "Legacy paper evidence contains withdrawn historical metrics and cannot "
        "be rebuilt as authoritative. See FINAL_IMPLEMENTATION_STATUS.md and "
        "PUBLICATION_REMAINING_WORK.md."
    )


if __name__ == "__main__":
    main()

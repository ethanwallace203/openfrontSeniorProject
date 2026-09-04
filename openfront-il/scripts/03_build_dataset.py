#!/usr/bin/env python
"""Thin CLI entry point — all logic lives in
src/openfront_il/features/dataset_builder.py. See scripts/README.md.
"""

from openfront_il.features.dataset_builder import build_dataset


def main() -> None:
    outputs = build_dataset()
    print(f"built {len(outputs)} processed dataset file(s):")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()

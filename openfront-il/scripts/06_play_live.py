#!/usr/bin/env python
"""Thin CLI entry point — drops a trained checkpoint into live, headless
singleplayer games against nations and tribes, and prints how it did.
Play logic lives in `src/openfront_il/agent/play.py`; the engine process
in `engine/bridge/src/liveGame.ts`. See scripts/README.md.

Usage:
    python scripts/06_play_live.py [--checkpoint PATH] [--map Onion] [--bots 4]
        [--difficulty Medium] [--games 1] [--seed S] [--greedy] [--temperature 1.0]
        [--max-steps 5000] [--no-save] [--verbose]
"""

import argparse
import logging
import zlib
from pathlib import Path

from openfront_il.agent.live_game import LiveGame
from openfront_il.agent.play import DEFAULT_CHECKPOINT, PolicyAgent, play_episode
from openfront_il.utils import paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--map", default="Onion")
    parser.add_argument("--bots", type=int, default=4)
    parser.add_argument("--difficulty", default="Medium")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", default=None, help="base seed; game i uses '<seed>-<i>'")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="log every intent sent")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s"
    )
    if not args.checkpoint.exists():
        print(f"no checkpoint found at {args.checkpoint} - run scripts/04_train_bc.py first")
        return

    agent = PolicyAgent.from_checkpoint(
        args.checkpoint,
        greedy=args.greedy,
        temperature=args.temperature,
        seed=None if args.seed is None else zlib.crc32(args.seed.encode()),
    )
    wins = 0
    with LiveGame(log_path=paths.AGENT_GAMES_DIR / "engine.log") as game:
        for i in range(args.games):
            result = play_episode(
                agent,
                game,
                map=args.map,
                seed=None if args.seed is None else f"{args.seed}-{i}",
                bots=args.bots,
                difficulty=args.difficulty,
                max_steps=args.max_steps,
                save=not args.no_save,
            )
            wins += result.won
            outcome = "WON" if result.won else ("survived" if result.survived else "eliminated")
            print(
                f"game {i + 1}/{args.games} [{result.game_id}] {outcome} at tick "
                f"{result.final_tick} ({result.steps} steps) - tiles peak={result.peak_tiles} "
                f"final={result.final_tiles}, intents sent={result.intents_sent}"
            )
            print("  actions:", dict(result.action_counts.most_common()))
            if result.record_path is not None:
                print("  record:", result.record_path)
    print(f"won {wins}/{args.games}")


if __name__ == "__main__":
    main()

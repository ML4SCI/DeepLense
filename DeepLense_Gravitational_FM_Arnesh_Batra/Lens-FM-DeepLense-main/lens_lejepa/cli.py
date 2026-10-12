"""Command-line interface.

    python -m lens_lejepa pretrain --config configs/pretrain/lens_lejepa.yaml
    python -m lens_lejepa finetune --config configs/downstream/classification.yaml --set shots=100
    python -m lens_lejepa evaluate --checkpoint runs/.../best.pt --test-root ../datasets/Model_III_test
    python -m lens_lejepa tasks
"""

from __future__ import annotations

import argparse

from .config import FinetuneConfig, PretrainConfig, load_config
from .utils import setup_logging, write_json


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="YAML file; any field not listed keeps its paper default.")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Override fields, e.g. --set epochs=1 shots=100")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="lens_lejepa", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    pretrain_parser = commands.add_parser("pretrain", help="self-supervised pretraining")
    _add_config_args(pretrain_parser)
    pretrain_parser.add_argument("--resume", help="last_pretrain.pt to continue from")

    finetune_parser = commands.add_parser("finetune", help="train a downstream task")
    _add_config_args(finetune_parser)

    evaluate_parser = commands.add_parser("evaluate", help="held-out evaluation of a downstream checkpoint")
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--test-root", required=True, help="e.g. ../datasets/Model_II_test")
    evaluate_parser.add_argument("--device", default="cuda")
    evaluate_parser.add_argument("--workers", type=int, default=8)
    evaluate_parser.add_argument("--output", help="write metrics to this JSON file")

    commands.add_parser("tasks", help="list registered downstream tasks")

    args = parser.parse_args(argv)
    setup_logging()

    # Imported lazily so ``--help`` stays fast.
    if args.command == "pretrain":
        from .engine import pretrain

        pretrain(load_config(PretrainConfig, args.config, args.set), resume=args.resume)
    elif args.command == "finetune":
        from .engine import finetune

        finetune(load_config(FinetuneConfig, args.config, args.set))
    elif args.command == "evaluate":
        from pathlib import Path

        from .engine import evaluate

        metrics = evaluate(args.checkpoint, args.test_root, args.device, workers=args.workers)
        if args.output:
            write_json(Path(args.output), metrics)
    elif args.command == "tasks":
        from .tasks import TASKS

        for name, cls in sorted(TASKS.items()):
            direction = "higher" if cls.higher_is_better else "lower"
            print(f"{name:18s} monitor={cls.monitor} ({direction} is better)  {cls.__doc__.strip().splitlines()[0] if cls.__doc__ else ''}")


if __name__ == "__main__":
    main()

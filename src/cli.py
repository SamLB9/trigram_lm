#!/usr/bin/env python3
"""
Single command-line entry point, mirrors the project README.

Examples
--------
# Train a character trigram LM
$ python -m src.cli train \
      --input  data/training.en \
      --output models/char.pkl \
      --model  char \
      --alpha  0.5

# Generate 500 characters
$ python -m src.cli generate \
      --model  models/char.pkl \
      --length 500

# Compute perplexity on held-out data
$ python -m src.cli perplexity \
      --model  models/char.pkl \
      --input  data/heldout.en
"""
from __future__ import annotations

import argparse
import pathlib
import sys

from .ngram_model import CharTrigramModel, WordTrigramModel

# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
MODELS = {"char": CharTrigramModel, "word": WordTrigramModel}


def _load(path: str | pathlib.Path):
    import pickle

    with open(path, "rb") as fh:
        return pickle.load(fh)


# --------------------------------------------------------------------- #
# Sub-commands
# --------------------------------------------------------------------- #
def _cmd_train(args: argparse.Namespace) -> None:
    cls = MODELS[args.model]
    model = cls(alpha=args.alpha)

    text = pathlib.Path(args.input).read_text(encoding="utf-8")
    if args.model == "char":
        model.train_from_text(text)
    else:
        model.train_from_text(text)

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"✓ saved model → {args.output}")


def _cmd_generate(args: argparse.Namespace) -> None:
    model = _load(args.model)
    gen = (
        model.generate_text(args.length, args.seed)
        if hasattr(model, "generate_text")
        else model.generate(args.length)
    )
    print(gen)


def _cmd_perplexity(args: argparse.Namespace) -> None:
    model = _load(args.model)
    text = pathlib.Path(args.input).read_text(encoding="utf-8")
    ppl = model.perplexity(list(text) if args.model_type == "char" else text.split())
    print(f"Perplexity: {ppl:.3f}")


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:  # noqa: D401
    parser = argparse.ArgumentParser(
        prog="trigram-lm",
        description="Train, sample, and evaluate smoothed trigram LMs",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ------------- train ------------- #
    ptrain = sub.add_parser("train", help="train a model")
    ptrain.add_argument("--input", "-i", required=True)
    ptrain.add_argument("--output", "-o", required=True)
    ptrain.add_argument("--model", "-m", choices=MODELS, default="char")
    ptrain.add_argument("--alpha", type=float, default=1.0)
    ptrain.set_defaults(func=_cmd_train)

    # ------------- generate ---------- #
    pgen = sub.add_parser("generate", help="sample from a model")
    pgen.add_argument("--model", "-m", required=True)
    pgen.add_argument("--length", "-l", type=int, default=300)
    pgen.add_argument("--seed", "-s", default=None)
    pgen.set_defaults(func=_cmd_generate)

    # ------------- perplexity -------- #
    pppl = sub.add_parser("perplexity", help="compute perplexity")
    pppl.add_argument("--model", "-m", required=True)
    pppl.add_argument("--input", "-i", required=True)
    pppl.add_argument(
        "--model-type",
        choices=["char", "word"],
        default="char",
        help="needed so we tokenise correctly",
    )
    pppl.set_defaults(func=_cmd_perplexity)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
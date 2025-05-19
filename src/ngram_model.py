"""
Reusable n-gram (default: trigram) language-model with Laplace smoothing.

The implementation is deliberately simple, no external ML frameworks
needed, but fully documented and unit-tested.
"""
from __future__ import annotations

import math
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

__all__ = ["CharTrigramModel", "WordTrigramModel"]


class _BaseNGramModel:
    """
    Generic additive-smoothed n-gram language model.

    Parameters
    ----------
    n
        Order of the model. We keep ``n = 3`` fixed in the public wrappers.
    alpha
        Add-α (Laplace) smoothing constant.
    """

    def __init__(self, n: int = 3, *, alpha: float = 1.0) -> None:
        if n < 2:
            raise ValueError("n-gram order must be ≥ 2")
        self.n = n
        self.alpha = alpha
        self._ngram_counts: Counter[Tuple[str, ...]] = Counter()
        self._context_counts: Counter[Tuple[str, ...]] = Counter()
        self._vocab: set[str] = set()
        self._probs: Dict[Tuple[str, ...], Dict[str, float]] | None = None

    # ------------------------------------------------------------------ #
    # Training utilities
    # ------------------------------------------------------------------ #
    def _update_counts(self, tokens: Sequence[str]) -> None:
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i : i + self.n])
            context = ngram[:-1]
            self._ngram_counts[ngram] += 1
            self._context_counts[context] += 1
            self._vocab.update(ngram)

    def _freeze_probs(self) -> None:
        """Convert raw counts to a smoothed conditional probability table."""
        V = len(self._vocab)
        probs: Dict[Tuple[str, ...], Dict[str, float]] = defaultdict(dict)
        for ngram, count in self._ngram_counts.items():
            context, target = ngram[:-1], ngram[-1]
            probs[context][target] = (count + self.alpha) / (
                self._context_counts[context] + self.alpha * V
            )
        self._probs = probs

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def train(self, tokens: Sequence[str]) -> None:
        """Estimate n-gram probabilities from tokens (already pre-processed)."""
        self._update_counts(tokens)
        self._freeze_probs()

    def save(self, path: str | Path) -> None:
        """Pickle the trained model to path."""
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    # pylint: disable=too-many-branches
    @classmethod
    def load(cls, path: str | Path) -> "_BaseNGramModel":
        """Load a pickled model."""
        with open(path, "rb") as fh:
            model = pickle.load(fh)
        if not isinstance(model, cls):
            raise TypeError(f"Pickle does not contain a {cls.__name__}")
        return model

    # ---------------------------- generation -------------------------- #
    def generate(
        self,
        length: int = 300,
        seed: Sequence[str] | None = None,
    ) -> List[str]:
        """
        Draw length tokens from the model, starting with seed context.

        The generator falls back to a random context if the current history
        was never seen during training, making generation robust even for
        tiny corpora.
        """
        if self._probs is None:
            raise RuntimeError("Model not trained yet")

        rng = random.Random()
        context = self._pick_start(seed)
        out: List[str] = list(context)

        while len(out) < length:
            dist = self._probs.get(context)
            if not dist:  # unseen context → restart
                context = self._pick_start(None)
                out.extend(context)
                continue

            tokens, weights = zip(*dist.items())
            next_tok = rng.choices(tokens, weights=weights)[0]
            out.append(next_tok)
            context = tuple(out[-(self.n - 1) :])

        return out[:length]

    def _pick_start(self, seed: Sequence[str] | None) -> Tuple[str, ...]:
        if seed is None:
            return random.choice(list(self._probs))  # type: ignore[arg-type]
        seed = tuple(seed)[-(self.n - 1) :]
        return seed if seed in self._probs else random.choice(list(self._probs))

    # ---------------------------- perplexity ------------------------- #
    def perplexity(self, tokens: Sequence[str]) -> float:
        """Compute test corpus perplexity (base e) with additive smoothing."""
        if self._probs is None:
            raise RuntimeError("Model not trained yet")

        V = len(self._vocab)
        log_prob = 0.0
        n_trigrams = 0

        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i : i + self.n])
            context, target = ngram[:-1], ngram[-1]
            prob = self._probs.get(context, {}).get(target)
            if prob is None:  # unseen → fall back to α / (cnt + α V)
                denom = self._context_counts.get(context, 0) + self.alpha * V
                prob = self.alpha / denom
            log_prob += math.log(prob)
            n_trigrams += 1

        return math.exp(-log_prob / n_trigrams) if n_trigrams else float("inf")


# ======================================================================#
#   Public, convenience subclasses
# ======================================================================#
class CharTrigramModel(_BaseNGramModel):
    """Character-level trigram with Laplace smoothing."""

    def __init__(self, *, alpha: float = 1.0) -> None:
        super().__init__(n=3, alpha=alpha)

    # thin wrappers keep the external API explicit and self-documenting
    def train_from_text(self, text: str) -> None:  # noqa: D401
        """Train on raw text (characters)."""
        super().train(list(text))

    def generate_text(self, length: int = 300, seed: str | None = None) -> str:
        tokens = super().generate(length, list(seed) if seed else None)
        return "".join(tokens)


class WordTrigramModel(_BaseNGramModel):
    """Word-level trigram with Laplace smoothing."""

    def __init__(self, *, alpha: float = 1.0) -> None:
        super().__init__(n=3, alpha=alpha)

    def train_from_text(self, text: str) -> None:  # noqa: D401
        super().train(text.split())

    def generate_text(
        self, length: int = 100, seed: Sequence[str] | None = None
    ) -> str:
        tokens = super().generate(length, seed)
        return " ".join(tokens)
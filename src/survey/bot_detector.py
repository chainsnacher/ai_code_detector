from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math
import re

import numpy as np
import pandas as pd

from .similarity import tfidf_nearest_neighbor_similarity


_WS_RE = re.compile(r"\s+")
_MULTI_PUNCT_RE = re.compile(r"([!?.;,])\1{2,}")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{6,}")


def _normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = _WS_RE.sub(" ", s)
    return s


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts: Dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * math.log(p, 2)
    return float(ent)


def _simple_sentences(text: str) -> List[str]:
    # Lightweight sentence split (avoid NLTK downloads for Streamlit Cloud)
    parts = re.split(r"[.!?]+\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


@dataclass
class ScoreOutput:
    risk_score: float
    flagged: bool
    label: str
    reasons: List[str]
    metrics: Dict[str, Any]


class SurveyBotDetector:
    """
    Text-only detector for survey bot / fake free-text responses.

    Design goals:
    - Explainable: returns reasons + measurable metrics
    - Fast: works for a few thousand rows in Streamlit Cloud
    - No heavy models (no torch/transformers)
    """

    def __init__(
        self,
        *,
        duplicate_similarity_threshold: float = 0.92,
        short_word_threshold: int = 4,
    ):
        self.duplicate_similarity_threshold = float(duplicate_similarity_threshold)
        self.short_word_threshold = int(short_word_threshold)

    def score_text(self, text: str, *, risk_threshold: float = 0.65) -> Dict[str, Any]:
        t = _normalize_text(text)
        metrics = self._compute_text_metrics(t)
        score, reasons = self._score_from_metrics(metrics)

        # Standalone text scoring can’t detect duplicates reliably; keep those reasons out here.
        score = float(np.clip(score, 0.0, 1.0))
        flagged = bool(score >= float(risk_threshold))
        label = "Flagged" if flagged else "OK"

        return ScoreOutput(
            risk_score=score,
            flagged=flagged,
            label=label,
            reasons=reasons,
            metrics=metrics,
        ).__dict__

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        *,
        text_col: str,
        risk_threshold: float = 0.65,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if text_col not in df.columns:
            raise ValueError(f"Column not found: {text_col}")

        out = df.copy()
        raw_texts = out[text_col].astype(str).fillna("").tolist()
        norm_texts = [_normalize_text(t) for t in raw_texts]

        # Exact duplicates
        norm_series = pd.Series(norm_texts, name="_norm_text")
        dup_sizes = norm_series.map(norm_series.value_counts()).astype(int)
        out["duplicate_group_size"] = dup_sizes.values

        # Near duplicates (TF-IDF cosine NN)
        # If dataset is too small or too uniform, TF-IDF can be degenerate; handle gracefully.
        nn_index = np.full(len(out), -1, dtype=int)
        nn_sim = np.zeros(len(out), dtype=float)
        try:
            sim_res = tfidf_nearest_neighbor_similarity(norm_texts)
            nn_index = sim_res.nearest_index
            nn_sim = sim_res.nearest_similarity
        except Exception:
            pass

        out["near_duplicate_index"] = nn_index
        out["near_duplicate_similarity"] = nn_sim

        # Base text metrics + scoring
        risk_scores: List[float] = []
        reasons_list: List[List[str]] = []

        for i, t in enumerate(norm_texts):
            metrics = self._compute_text_metrics(t)
            score, reasons = self._score_from_metrics(metrics)

            # Duplicate penalties (strong bot signal in surveys)
            if int(out.at[out.index[i], "duplicate_group_size"]) >= 3:
                score += 0.35
                reasons.append("many_exact_duplicates")
            elif int(out.at[out.index[i], "duplicate_group_size"]) == 2:
                score += 0.20
                reasons.append("exact_duplicate")

            if float(out.at[out.index[i], "near_duplicate_similarity"]) >= self.duplicate_similarity_threshold:
                score += 0.25
                reasons.append("near_duplicate")

            score = float(np.clip(score, 0.0, 1.0))
            risk_scores.append(score)
            reasons_list.append(sorted(set(reasons)))

        out["risk_score"] = np.array(risk_scores, dtype=float)
        out["flagged"] = out["risk_score"] >= float(risk_threshold)
        out["label"] = np.where(out["flagged"], "Flagged", "OK")
        out["reasons"] = [";".join(r) if r else "" for r in reasons_list]

        # Report
        total = int(len(out))
        flagged = int(out["flagged"].sum())
        exact_dup_rows = int((out["duplicate_group_size"] >= 2).sum())

        reason_counts: Dict[str, int] = {}
        for rs in reasons_list:
            for r in rs:
                reason_counts[r] = reason_counts.get(r, 0) + 1
        top_reasons = sorted(reason_counts.items(), key=lambda x: (-x[1], x[0]))[:15]

        report = {
            "total": total,
            "flagged": flagged,
            "flag_rate": (flagged / total) if total else 0.0,
            "exact_duplicate_rows": exact_dup_rows,
            "top_reasons": top_reasons,
            "risk_threshold": float(risk_threshold),
        }

        return out, report

    def _compute_text_metrics(self, text: str) -> Dict[str, Any]:
        t = text or ""
        words = _tokenize_words(t)
        n_words = len(words)
        n_chars = len(t)

        unique_words = len(set(words))
        unique_ratio = unique_words / max(n_words, 1)

        # Character category ratios
        digits = sum(1 for c in t if c.isdigit())
        letters = sum(1 for c in t if c.isalpha())
        spaces = sum(1 for c in t if c.isspace())
        specials = max(n_chars - digits - letters - spaces, 0)
        digit_ratio = digits / max(n_chars, 1)
        alpha_ratio = letters / max(n_chars, 1)
        special_ratio = specials / max(n_chars, 1)

        entropy = _shannon_entropy(t)

        # Repetition indicators
        repeated_char = bool(_REPEATED_CHAR_RE.search(t))
        multi_punct = bool(_MULTI_PUNCT_RE.search(t))

        # Sentence-level uniformity proxy
        sents = _simple_sentences(t)
        sent_lens = [len(_tokenize_words(s)) for s in sents] if sents else []
        if len(sent_lens) >= 2 and float(np.mean(sent_lens)) > 0:
            sent_len_cv = float(np.std(sent_lens) / (np.mean(sent_lens) + 1e-9))
        else:
            sent_len_cv = 0.0

        # N-gram template proxy: repeated bigrams / trigrams
        bigrams = list(zip(words, words[1:])) if len(words) >= 2 else []
        trigrams = list(zip(words, words[1:], words[2:])) if len(words) >= 3 else []
        bigram_counts: Dict[Tuple[str, str], int] = {}
        for bg in bigrams:
            bigram_counts[bg] = bigram_counts.get(bg, 0) + 1
        trigram_counts: Dict[Tuple[str, str, str], int] = {}
        for tg in trigrams:
            trigram_counts[tg] = trigram_counts.get(tg, 0) + 1

        max_bigram_frac = (max(bigram_counts.values()) / max(len(bigrams), 1)) if bigrams else 0.0
        max_trigram_frac = (max(trigram_counts.values()) / max(len(trigrams), 1)) if trigrams else 0.0

        return {
            "n_chars": n_chars,
            "n_words": n_words,
            "unique_word_ratio": float(unique_ratio),
            "digit_ratio": float(digit_ratio),
            "alpha_ratio": float(alpha_ratio),
            "special_ratio": float(special_ratio),
            "shannon_entropy": float(entropy),
            "repeated_char_run": bool(repeated_char),
            "excessive_punctuation": bool(multi_punct),
            "sentence_len_cv": float(sent_len_cv),
            "max_bigram_fraction": float(max_bigram_frac),
            "max_trigram_fraction": float(max_trigram_frac),
        }

    def _score_from_metrics(self, m: Dict[str, Any]) -> Tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []

        n_words = int(m.get("n_words", 0))
        n_chars = int(m.get("n_chars", 0))
        uniq = float(m.get("unique_word_ratio", 0.0))
        special_ratio = float(m.get("special_ratio", 0.0))
        alpha_ratio = float(m.get("alpha_ratio", 0.0))
        digit_ratio = float(m.get("digit_ratio", 0.0))
        ent = float(m.get("shannon_entropy", 0.0))
        cv = float(m.get("sentence_len_cv", 0.0))
        max_bg = float(m.get("max_bigram_fraction", 0.0))
        max_tg = float(m.get("max_trigram_fraction", 0.0))

        if n_words <= self.short_word_threshold:
            score += 0.25
            reasons.append("too_short")

        if n_chars >= 20 and special_ratio > 0.28:
            score += 0.25
            reasons.append("high_symbol_ratio")

        if n_chars >= 20 and alpha_ratio < 0.55 and (digit_ratio + special_ratio) > 0.45:
            score += 0.20
            reasons.append("low_alpha_ratio")

        # Very low entropy often indicates repetitive/template/gibberish; very high can indicate noise.
        if n_chars >= 25 and ent < 3.3:
            score += 0.20
            reasons.append("low_entropy")
        elif n_chars >= 25 and ent > 5.2 and alpha_ratio < 0.70:
            score += 0.15
            reasons.append("noisy_text_entropy")

        if bool(m.get("repeated_char_run", False)):
            score += 0.25
            reasons.append("repeated_characters")

        if bool(m.get("excessive_punctuation", False)):
            score += 0.15
            reasons.append("excessive_punctuation")

        # Low unique ratio suggests templated spam.
        if n_words >= 12 and uniq < 0.55:
            score += 0.18
            reasons.append("low_word_diversity")

        # Strong repeated n-grams suggest templates.
        if n_words >= 15 and (max_bg > 0.18 or max_tg > 0.14):
            score += 0.22
            reasons.append("repeated_ngrams")

        # Uniform sentence lengths can be an LLM/spam signal (very low CV).
        if n_words >= 25 and 0.0 <= cv < 0.20:
            score += 0.12
            reasons.append("uniform_sentence_lengths")

        return score, reasons


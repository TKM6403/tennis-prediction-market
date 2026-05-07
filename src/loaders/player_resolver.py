"""
player_resolver.py

PlayerResolver: maps player names from external sources (Kalshi titles,
Polymarket questions, etc.) to TML's canonical (player_id, full_name) pair.

The motivating problem
----------------------
Kalshi parses match titles into (player_a, player_b) where player_a is a
full name but player_b is the last-name token from the "X vs Y" clause.
TML carries full names ("Damir Dzumhur"), so naive last-name string
matching fails: "Dzumhur" != "Damir Dzumhur" → every diff feature comes
out NaN. Worse, last-name-only matching is silently wrong in shared-last-
name cases (Juan Manuel Cerundolo vs Francisco Cerundolo, the brothers).

Sackmann TML data carries `winner_id` / `loser_id` — stable integer ATP
ids that survive name spelling changes and disambiguate brothers. Once
exposed by `TMLMatchLoader.normalize()`, this class consumes them to
build a name → (id, full_name) lookup table.

Resolution rules (in order)
---------------------------
1. **Exact full-name match** ("Luciano Darderi" → unique id).
2. **Last-name only**: pick the most-frequent player_id with that last name.
3. **Last-name with hint** (tournament + date): TODO; not implemented yet.
   Will narrow to players who actually appeared at that event.
4. **No match**: return (None, name unchanged) so the caller can decide
   whether to skip or proceed without an id.

Ambiguity is logged at WARN; callers should treat ambiguous resolutions
as suspect (especially for shared last names).

Usage
-----
    from src.loaders.player_resolver import PlayerResolver

    resolver = PlayerResolver(tml_df)            # built once per session
    pid, full = resolver.resolve("Dzumhur")     # → (105357, "Damir Dzumhur")
    pid, full = resolver.resolve("Cerundolo")   # → ambiguous, picks the
                                                #   most-frequent Cerundolo
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def _normalize_token(s: str) -> str:
    """Lowercase + strip non-alpha (keeping hyphens for "Auger-Aliassime")."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"[^a-z\-]", "", s.lower())


def _last_name_token(full_name: str) -> str:
    """Take the final whitespace-delimited token from a name; normalize."""
    if not isinstance(full_name, str) or not full_name.strip():
        return ""
    return _normalize_token(full_name.strip().split()[-1])


class PlayerResolver:
    """
    Build a lookup index from a normalized TML DataFrame, then resolve
    arbitrary name strings to (player_id, canonical_full_name).

    Expects `tml_df` to have these columns:
        player_a, player_b, winner_id, loser_id  (output of TMLMatchLoader.normalize)

    Falls back gracefully if winner_id/loser_id are missing — in that case
    the resolver returns (None, full_name) for every hit, which still buys
    you the canonical name upgrade even without ids.
    """

    def __init__(self, tml_df: pd.DataFrame):
        # Per-player observation counts, keyed by id (or by full name if id missing).
        # We keep the most common full_name per id as the canonical spelling
        # (TML occasionally varies — e.g. "Auger-Aliassime" vs "AugerAliassime").
        full_name_counts: dict = {}        # id_or_name -> Counter[full_name]
        last_name_to_ids: dict = {}        # last_name_token -> Counter[id_or_name]
        full_name_to_ids: dict = {}        # normalized_full_name -> Counter[id_or_name]
        first_name_tokens: set = set()    # set of normalized first-name tokens
        n_rows = 0

        has_ids = "winner_id" in tml_df.columns and "loser_id" in tml_df.columns
        if not has_ids:
            logger.warning(
                "PlayerResolver: TML has no winner_id/loser_id columns — "
                "falling back to name-only resolution. Update "
                "TMLMatchLoader.normalize() to expose Sackmann ids."
            )

        for _, row in tml_df.iterrows():
            for side in ("a", "b"):
                name = row.get(f"player_{side}")
                if not isinstance(name, str) or not name.strip():
                    continue
                if has_ids:
                    pid = row.get("winner_id" if side == "a" else "loser_id")
                    # Some TML rows have NaN id (rare, non-tour level)
                    if pd.isna(pid):
                        continue
                    # Sackmann ids are usually integer but futures/ITF carry
                    # prefixed strings like "B752". Keep as-is — string keys
                    # work fine for dict lookups and downstream comparisons.
                    if isinstance(pid, float) and pid.is_integer():
                        pid = int(pid)
                    key = pid
                else:
                    key = name.strip()

                full_name_counts.setdefault(key, Counter())[name.strip()] += 1
                last = _last_name_token(name)
                if last:
                    last_name_to_ids.setdefault(last, Counter())[key] += 1
                fnorm = _normalize_token(name.replace(" ", ""))
                if fnorm:
                    full_name_to_ids.setdefault(fnorm, Counter())[key] += 1
                # First-name registry: the leading whitespace-delimited token
                # of each player's full name. Used during last-name fallback
                # to validate that an input's leading token is plausibly a
                # first name and not, e.g., the surname "Pucinelli" appearing
                # at the front of a name string Kalshi happens to send.
                first_tok = _normalize_token(name.strip().split()[0])
                if first_tok:
                    first_name_tokens.add(first_tok)
                n_rows += 1

        self._has_ids = has_ids
        self._canonical_name = {
            k: counter.most_common(1)[0][0] for k, counter in full_name_counts.items()
        }
        self._last_name_to_ids = last_name_to_ids
        self._full_name_to_ids = full_name_to_ids
        self._first_name_tokens = first_name_tokens

        logger.info(
            f"PlayerResolver indexed {len(self._canonical_name):,} unique "
            f"players from {n_rows:,} TML player-rows "
            f"(ids={'yes' if has_ids else 'no'})"
        )

    # ------------------------------------------------------------------ #

    def resolve(self, name: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Resolve `name` to (player_id, canonical_full_name).

        Returns (None, None) when the input is empty/non-string;
        (None, name) when no TML match found OR result is ambiguous
        (caller should skip the bet);
        (id, full_name) when resolved (id is None if TML has no ids).

        Resolution order:
          1. Exact full-name match.
          2. Multi-token input with multi-candidate last name → first-name
             token-overlap disambiguation. Refuse to guess if still tied.
          3. Single-token input → most-frequent player with that last name
             (legacy Kalshi-last-name-only behavior).
        """
        if not isinstance(name, str) or not name.strip():
            return (None, None)
        name = name.strip()

        # 1. Exact full-name match (case-insensitive, whitespace-insensitive)
        fnorm = _normalize_token(name.replace(" ", ""))
        candidates = self._full_name_to_ids.get(fnorm)
        if candidates:
            return self._pick(candidates)

        # 2. Last-name lookup
        last = _last_name_token(name)
        if not last:
            return (None, name)
        last_candidates = self._last_name_to_ids.get(last)
        if not last_candidates:
            return (None, name)

        # Single candidate: trivially correct.
        if len(last_candidates) == 1:
            return self._pick(last_candidates)

        # Multiple candidates: require the input's first-name token to
        # literally appear in the candidate's canonical name. Picking by
        # frequency or by partial token-overlap is unsafe when the input
        # is a player TML has never seen — we'd silently match the wrong
        # person (e.g. "Raphael Pucinelli de Almeida" sneaks into
        # "Matheus Pucinelli de Almeida" because all the last tokens align).
        input_tokens_list = [t for t in (_normalize_token(x) for x in name.split()) if t]
        if len(input_tokens_list) >= 2:
            first_token = input_tokens_list[0]
            # Guard against Kalshi sending compound surnames like "Pucinelli
            # De Almeida" (no first name): require `first_token` to actually
            # be a known first name in TML. Otherwise we'd treat the surname
            # itself as a discriminator and silently match the wrong person.
            if first_token not in self._first_name_tokens:
                cand_names = [self._canonical_name.get(k, str(k))
                              for k in list(last_candidates)[:3]]
                logger.warning(
                    f"PlayerResolver: '{name}' — leading token '{first_token}' "
                    f"is not a known first name; "
                    f"{len(last_candidates)} last-name candidates "
                    f"[{', '.join(cand_names)}]. Returning unresolved."
                )
                return (None, name)
            valid = []
            for key, freq in last_candidates.items():
                cano = self._canonical_name.get(key, str(key))
                cano_tokens = {_normalize_token(t) for t in cano.split() if t.strip()}
                cano_tokens.discard("")
                if first_token in cano_tokens:
                    valid.append((key, freq, cano))
            if len(valid) == 1:
                key, _, _ = valid[0]
                full = self._canonical_name.get(key, str(key))
                pid = key if self._has_ids else None
                return (pid, full)
            if len(valid) > 1:
                # First+last name shared by multiple players (rare — e.g.
                # generations of the same family). Refuse to guess.
                cand_str = ", ".join(c[2] for c in valid[:3])
                logger.warning(
                    f"PlayerResolver: '{name}' ambiguous — {len(valid)} "
                    f"candidates share first+last name [{cand_str}]. "
                    f"Returning unresolved."
                )
                return (None, name)
            # No candidate has this first name → input player is not in TML.
            cand_names = [self._canonical_name.get(k, str(k))
                          for k in list(last_candidates)[:3]]
            logger.warning(
                f"PlayerResolver: '{name}' — first name '{first_token}' "
                f"matches none of {len(last_candidates)} last-name candidates "
                f"[{', '.join(cand_names)}]. Returning unresolved."
            )
            return (None, name)

        # Single-token input (e.g. Kalshi's last-name-only player_b): fall
        # back to most-frequent. Logging unchanged from the legacy version.
        top_two = last_candidates.most_common(2)
        logger.info(
            f"PlayerResolver: '{name}' matched {len(last_candidates)} "
            f"players by last name; picked most-frequent "
            f"({top_two[0][0]}: {top_two[0][1]} rows vs "
            f"{top_two[1][0]}: {top_two[1][1]})"
        )
        return self._pick(last_candidates)

    def _pick(self, candidates: Counter) -> Tuple[Optional[int], str]:
        """Return (id_or_None, canonical_full_name) for the most-frequent key."""
        key = candidates.most_common(1)[0][0]
        full = self._canonical_name.get(key, str(key))
        # When ids are present, the dict key IS the id (int or string e.g. "B752").
        # When ids are absent the key is the name string and we return id=None.
        pid = key if self._has_ids else None
        return (pid, full)

# PlayerResolver audit — 9 "failing" Kalshi IDs

## Summary

The 9 IDs flagged this week (C0BB, J0D2, G0AO, P0HT, SY50, S0TI, S0IV, H0J1, K0AB)
are **not actually failing** — each resolves cleanly to a unique TML player with
4-67+ historical matches. The 187 dropped markets they appear in are caused by
**(a) their opponents** not resolving (116 `missing_player_id` rows) and
**(b) tournament-name mismatches** (64 `tournament_not_in_tml` rows — Bengaluru
2/3, Cervia). Verdict: this is a **mix of (b) TML coverage gaps for a small
cluster of Challenger-circuit specialists and (c) Kalshi/TML name+tournament
metadata mismatches** — not (a) a resolver normalization bug for the 9 named
players.

## Per-ID findings

Each ID maps 1:1 to one Kalshi human across all dropped rows (no ID reuse).

| Kalshi ID | Kalshi name                    | TML candidate                  | Resolver verdict                | Drops |
|-----------|--------------------------------|--------------------------------|---------------------------------|-------|
| C0BB      | Tommaso Compagnucci            | Tommaso Compagnucci (7 m)      | Resolves fine                   | 34    |
| J0D2      | Maximus Jones                  | Maximus Jones (63 m)           | Resolves fine                   | 17    |
| G0AO      | Alastair Gray                  | Alastair Gray (53 m)           | Resolves fine                   | 34    |
| P0HT      | Jack Pinnington Jones          | Jack Pinnington Jones (67 m)   | Resolves fine                   | 19    |
| SY50      | Nitin Kumar Sinha              | Nitin Kumar Sinha (2 m)        | Resolves fine                   | 14    |
| S0TI      | Philip Sekulic                 | Philip Sekulic (103 m)         | Resolves fine                   | 40    |
| S0IV      | Paulo Andre Saraiva Dos Santos | Paulo Andre Saraiva D.S. (17 m)| Resolves fine                   | 9     |
| H0J1      | Naoya Honda                    | Naoya Honda (8 m)              | Resolves fine                   | 24    |
| K0AB      | Adil Kalyanpur                 | Adil Kalyanpur (4 m)           | Resolves fine                   | 10    |

**The real failure is on the opposite side of the ticket.** 116/187 drops are
`missing_player_id` where the OPPONENT's `player_id` is `None/NaN`. Top
unresolved opponents (with TML reality check):

- `Sasikumar Mukund` (14×) → **TML has him as "Mukund Sasikumar" (125 matches).**
  Kalshi-vs-TML first/last-name order is swapped. Last-token "Mukund" finds 0
  TML surnames; resolver returns `None`.
- `Aradhya Kshitij` (6×), `Tomohiro Masabayashi` (4×), `Alessio Balestrieri` (4×),
  `Volodymyr Iakubenko` (3×), `Daniel Bagnolini` (3×), `Avcibasi` (4×) — **not
  in TML** (Challenger/ITF-only juniors). Genuine coverage gap.
- `Dalla Valle` (6×) → **Enrico Dalla Valle (DG80) is in TML** but the resolver's
  "leading token must be a known first name" guard rejects "Dalla" (it's a
  compound-surname particle, not a first name).

## Root-cause buckets

**A. Tournament-name mismatch (64 drops, 34%)** — "Bengaluru 2", "Bengaluru 3",
"Cervia" are not in the TML tournament index. This is not a resolver issue at
all; it's tournament-string normalization downstream.

**B. TML genuinely missing the opponent (~46 drops, 25%)** — Kshitij, Masabayashi,
Balestrieri, Avcibasi, Iakubenko, Bagnolini, Anavatan. Challenger-Q-only
specialists who never appeared in tour-level or main-draw Challenger TML rows.

**C. Kalshi sends first/last swapped vs TML (~28 drops, 15%)** — Sasikumar Mukund
(Kalshi) vs Mukund Sasikumar (TML). Last-token strategy can't recover this.

**D. Compound-surname guard too strict (~6 drops, 3%)** — "Dalla Valle" rejected
because `dalla` isn't in `_first_name_tokens`. Same risk exists for "Bar Biryukov",
"Dos Santos", "Pucinelli De Almeida" when Kalshi sends surname-only.

**E. Low coverage on a resolved player (3 drops, 2%)** — S0IV (Saraiva Dos Santos)
has 12 prior matches at Istanbul vs the min-15 gate. Working as designed.

## Suggested fixes (surfacing only)

1. **Add a reversed-name pass to PlayerResolver** — when last-token lookup fails,
   try treating the FIRST token as the surname. Recovers Sasikumar Mukund + likely
   others. (1h, ~14-20 drops/week recovered)
2. **Whitelist compound-surname particles ("Dalla","Dos","De","Van","Da","Del")
   to bypass the first-name guard** when followed by a multi-token rest.
   Recovers Dalla Valle and prevents future "Dos Santos"-style drops.
   (1h, ~6-10 drops/week)
3. **Normalize tournament-name suffixes** — strip trailing " 2"/" 3"/"
   Qualification" before TML lookup, since TML stores the canonical event name
   without per-week numbering. Biggest single ROI here. (half-day, ~50-60
   drops/week, by far the highest yield)
4. **Accept the TML coverage gap for genuinely-absent Challenger-Q players** —
   no resolver change will fix Avcibasi/Kshitij/Masabayashi/Bagnolini. Either
   accept those drops as inventory we cannot price, or ingest a deeper feed
   (Sackmann's ITF/Challenger-Q files) — full-day, ~40-50 drops/week if
   ingestion succeeds.

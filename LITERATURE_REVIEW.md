# Literature Review — Tennis Prediction & Probability Calibration

**Audience:** high-schoolers / early-ML readers. The goal of this doc is
not to be exhaustive — it's to explain *why* our model looks the way it
does, in plain language, with the supporting papers cited so you can go
deeper if you want.

---

## The big picture in one paragraph

We want to bet on tennis matches on Kalshi (a prediction market). To do
that, we need to know the *true probability* a player wins. If our
probability is higher than the price Kalshi is selling at, we buy. To
figure out our probability, we train a **logistic regression** on
historical match data. The papers below answer three questions:

1. Why logistic regression and not something fancier?
2. How do we know our probabilities are *trustworthy*?
3. What's the actual evidence this kind of model makes money?

---

## The papers (in the order you should read them)

### 1. Walsh & Joshi (2024) — *the* paper for our project

**"Machine learning for sports betting: should model selection be based
on accuracy or calibration?"** *Machine Learning with Applications*,
June 2024.

**One-sentence summary:** Same models, same NBA data — picking the model
by **calibration** made +34.7% return. Picking by **accuracy** lost
35.2%. So model "accuracy" is the wrong number to chase.

**Why we cite it:** This is the modern, peer-reviewed proof that the
metric you optimize for matters more than the model itself. It's the
reason our `train.py` uses log-loss (a calibration-style metric) to
pick its best `C` instead of accuracy.

> "Kelly betting only works with a well-calibrated model."
> — Walsh & Joshi

[arXiv link](https://arxiv.org/abs/2303.06021v4)

---

### 2. Niculescu-Mizil & Caruana (2005) — "why logistic regression?"

**"Predicting Good Probabilities With Supervised Learning."** ICML 2005.

**One-sentence summary:** They tested every common ML model on
calibration. **Logistic regression and neural nets came out
well-calibrated by default.** SVMs and boosted trees did NOT — they
need extra fixing afterward.

**Why we cite it:** This is the empirical reason we don't reach for
XGBoost first. Logistic regression hands us calibrated probabilities
basically for free; tree-ensembles would force us to add a calibration
step on top, with more knobs and more ways to overfit.

[paper PDF](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf)

---

### 3. Kull, Silva Filho & Flach (2017) — Beta calibration

**"Beta calibration: a well-founded and easily implemented improvement
on logistic calibration for binary classifiers."** AISTATS 2017.

**One-sentence summary:** A new way to "fix" probabilities after the
fact. Better than the older Platt-scaling and isotonic methods when you
have small-to-medium data (which we do — ~5k tennis matches/year).

**Why we cite it:** If we ever want to *recalibrate* our LR output
(which we may, especially across surfaces), Beta calibration is the
modern default. It's a candidate to compare against the `isotonic` we
currently use.

---

### 4. Buhamra, Groll & Brunner (2024) — tennis-specific, recent

**"Statistical enhanced learning for modeling and prediction tennis
matches at Grand Slam tournaments."** *Journal of Sports Analytics*,
2024.

**One-sentence summary:** Tested 21 model combos (logistic regression,
GAMs, random forests) on 5,013 Grand Slam matches (2011–2022). They
evaluate on **Brier score** and **Bernoulli likelihood** — not
accuracy. Best features were **Elo rating**, **distance from age 30**,
and **ATP ranking points**.

**Why we cite it:** Direct confirmation that for tennis specifically,
LR with the right features is competitive with random forests and
beats raw-rank baselines. Their feature ideas (age curves, Elo) are
ones we can borrow.

[arXiv link](https://arxiv.org/abs/2502.01613)

---

### 5. Kovalchik (2016) — the tennis-modeling reference

**"Searching for the GOAT of tennis win prediction."** *Journal of
Quantitative Analysis in Sports*, 2016.

**One-sentence summary:** Compared Elo, Bradley-Terry, and logistic
regression for predicting tennis match winners. She **scores everything
on log loss**, not accuracy. Her takeaway: simple models with good
features beat fancy models with bad features.

**Why we cite it:** Proves the academic consensus on tennis is that
log-loss is the right scorer, and that LR-style models are the right
starting point.

---

### 6. Klaassen & Magnus (2003) — the foundational paper

**"Forecasting the winner of a tennis match."** *European Journal of
Operational Research*, 2003.

**One-sentence summary:** First serious attempt to model tennis as a
probability problem from point-by-point statistics. Showed that even a
simple model on serve stats and rankings could match bookmakers.

**Why we cite it:** Historical — this is where the field starts. Worth
knowing exists; not worth re-reading unless you're curious about the
roots.

---

### 7. Vandeghen & Louppe (2024) — survey paper

**"A Systematic Review of Machine Learning in Sports Betting:
Techniques, Challenges, and Future Directions."** arXiv 2410.21484.

**One-sentence summary:** Surveys 100+ recent papers across tennis,
NBA, soccer. Confirms the calibration-over-accuracy consensus, and
reviews Kelly sizing.

**Why we cite it:** Good "what else is out there" reading. Skim once,
then move on.

[arXiv link](https://arxiv.org/abs/2410.21484)

---

## Our proposed model — diagrammed

The whole pipeline, in one picture:

```
┌────────────────────────────────────────────────────────────────────┐
│                        RAW DATA SOURCES                            │
│                                                                    │
│   TML (Tennis ML) match results 2018-2024                          │
│   + tennis-data.co.uk (per-match dates)                            │
│   + Kalshi candlestick prices (for backtest only)                  │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                FEATURE ENGINEERING (per match)                     │
│                                                                    │
│   For each match, look ONLY at data BEFORE that match's date.      │
│   Compute 15 features:                                             │
│                                                                    │
│   • rank_ratio              (ATP rank A / ATP rank B)              │
│   • surface_win_rate_diff   (52-week win rate on this surface)     │
│   • recent_form_diff        (last 10 matches won)                  │
│   • fatigue_diff_21d / 28d  (minutes played recently)              │
│   • days_rest_diff          (days since last match)                │
│   • h2h_surface_diff        (head-to-head on this surface)         │
│   • ace_rate_diff           (aces / service points)                │
│   • df_rate_diff            (double faults / service points)       │
│   • serve_dominance_diff    (probability of winning a serve point) │
│   • return_dominance_diff   (how tough you make opponents serve)   │
│   • first_in_pct_diff                                              │
│   • first_won_pct_diff                                             │
│   • second_won_pct_diff                                            │
│   • height_diff                                                    │
│                                                                    │
│   ★ "diff" = player_A's value minus player_B's value               │
│   ★ The strict no-lookahead rule is the most important rule        │
│     in this whole project.                                         │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                  TIME-BASED SPLIT                                  │
│                                                                    │
│      ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│      │ TRAIN        │  │ VAL          │  │ TEST         │          │
│      │ 2021–2023    │  │ 2024         │  │ 2025         │          │
│      │ (~15k matches│  │ (~5k matches)│  │ (~5k matches)│          │
│      └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                    │
│   We split by DATE, not randomly. Why? Because betting tomorrow    │
│   means we never get to peek at the future. Random shuffles let    │
│   future data leak into training — that's the #1 way ML            │
│   models lie to you about how good they are.                       │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│              MODEL TRAINING — LOGISTIC REGRESSION                  │
│                                                                    │
│   Pipeline:                                                        │
│      [impute missing] → [scale features] →                         │
│      [logistic regression] → [calibrate w/ isotonic regression]    │
│                                                                    │
│   What logistic regression does:                                   │
│      P(A wins) = sigmoid( w₁·f₁ + w₂·f₂ + ... + w₁₅·f₁₅ + b )      │
│                                                                    │
│      Each weight wᵢ is learned. Positive wᵢ → "this feature        │
│      makes A more likely to win." Negative wᵢ → opposite.          │
│                                                                    │
│   We grid-search C (regularization strength) on the VAL set,       │
│   picking the C that minimizes LOG LOSS. This is the               │
│   Walsh & Joshi recommendation: select on calibration, not         │
│   accuracy.                                                        │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                        EVALUATION                                  │
│                                                                    │
│   On the TEST set (which the model has never seen), report:        │
│                                                                    │
│   • Log loss        — primary metric. Penalizes overconfident      │
│                       wrong predictions.                           │
│   • Brier score     — mean squared error between predicted prob    │
│                       and 0/1 outcome. Lower = better calibrated.  │
│   • Reliability     — visual sanity check: when we say 70%, do     │
│     diagram          the players actually win 70% of the time?     │
│   • Accuracy        — last priority. Just a sanity number.         │
│                                                                    │
│   Output files:                                                    │
│      data/processed/model_augmented.pkl      ← saved model         │
│      data/processed/reliability_*.png        ← calibration plots   │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                  BETTING (the whole point)                         │
│                                                                    │
│   For each upcoming match:                                         │
│      theo  = model.predict_proba(features)                         │
│      price = Kalshi YES price (also a probability, e.g. 0.62)      │
│      edge  = theo - price                                          │
│                                                                    │
│   If edge > +threshold  →  buy YES                                 │
│   If edge < -threshold  →  buy NO                                  │
│   Otherwise              →  skip                                   │
│                                                                    │
│   Bet sizing: Kelly criterion (or 1/4-Kelly to be safe).           │
│   ★ Kelly only works if `theo` is well-calibrated.                 │
└────────────────────────────────────────────────────────────────────┘
```

---

## Why this design, in plain English

**Why logistic regression and not a fancy neural net or XGBoost?**
Three reasons:

1. **Logistic regression outputs probabilities natively.** Other models
   output a "score" that you have to massage into a probability, and
   the massaging is often wrong (Niculescu-Mizil & Caruana 2005).
2. **It can't overfit easily.** With ~15 features and ~15,000 training
   matches, a neural net would memorize the training data and lie to
   us on the test set. LR has so few knobs that it's forced to learn
   actual signal.
3. **It's interpretable.** When the model bets on Sinner, we can read
   the coefficients and see exactly *why* — "high serve dominance,
   good recent form, opponent fatigued." When it loses money, we can
   debug. With a neural net we just shrug.

**Why log-loss as the metric, not accuracy?** Walsh & Joshi (2024)
showed empirically that picking the model by accuracy loses money;
picking by calibration makes money. And for betting against a market
that's ALSO quoting probabilities, you need YOUR probabilities to be
correct, not just on the right side of 50%.

**Why time-based split instead of random?** If you shuffle randomly,
you train on 2024 matches and test on 2022 — that's like getting
tomorrow's newspaper before placing today's bet. Useless for the real
problem.

**What's the next thing to read after this doc?** Walsh & Joshi (2024)
is the highest-leverage paper. Read it once and the rest of this
project's design choices will make immediate sense.

# Sample-size methodology for the evaluation set

Issue: [#7](https://github.com/news-trackers/nlp-workspace/issues/7) (parent [#5](https://github.com/news-trackers/nlp-workspace/issues/5))
Source: [`news-trackers/news-data`](https://huggingface.co/datasets/news-trackers/news-data), `data/train.parquet`, revision `7dee9570f394f6b945a9e2b689888feaf4777962`
Profiling that these numbers rest on: [`01_explore_news_data.ipynb`](01_explore_news_data.ipynb)

The sample size below is derived from a target precision, not picked as a round number.

## 1. Population

All 8,469 posts in the pinned revision, minus rows that cannot be labeled or would bias the estimate:

| Excluded | Count | Why |
|---|---|---|
| whitespace-only content | 133 | nothing to label |
| exact duplicates (after signature stripping) | 195 | would be labeled twice and inflate agreement |
| fewer than 5 words | 208 | too short to carry sentiment or entities |
| no letters at all (emoji / digits only) | 135 | no text for either task |

These sets overlap, so the eligible population is **N ≈ 7,950** (the notebook recomputes the exact figure).

The dataset is being restructured into `/data/<date-range>/telegram.parquet` and `twitter.parquet` (#23), so these counts will be recomputed against the new layout. Section 2 shows why that does not move the sample size.

## 2. Size

Cochran's formula for a proportion, with the finite-population correction:

```
n0 = z² · p(1-p) / e²
n  = n0 / (1 + (n0 - 1) / N)
```

- `z = 1.96` — 95% confidence
- `p = 0.5` — worst case; no prior estimate of per-class accuracy
- `e` — margin of error on a measured accuracy or F1

| Margin `e` | n0 | n after FPC (N = 7,950) |
|---|---|---|
| ±3% | 1,067 | 941 |
| **±5%** | **384** | **367** |
| ±7% | 196 | 192 |
| ±10% | 96 | 95 |

**Chosen: ±5%, so n = 367, rounded up to 400.** The extra 33 posts absorb rows discarded during labeling (unreadable, wrong language, pure link).

**The size barely depends on N.** The finite-population correction only bites for small populations, so a larger corpus — once X posts and more days are added — does not change the answer:

| N | n at ±5% |
|---|---|
| 7,950 | 367 |
| 20,000 | 377 |
| 100,000 | 383 |

Whatever the corpus grows to, the sample stays between 367 and 384. Only the allocation in section 3 has to be recomputed.

Why not ±3%: 941 posts × two labeling tasks is roughly 60 hours of annotation and does not fit before the 2026-09-29 milestone. ±5% distinguishes a model at 0.80 from one at 0.90, which is the decision #11 and #16 actually have to make. Why not ±10%: at that width two candidate models would not be separable.

## 3. Allocation

Stratified by language (the script heuristic from section 6 of the profiling notebook), proportional, with a floor of 25 so the small strata still yield a usable per-stratum number:

| Stratum | Posts | Share | Sample |
|---|---|---|---|
| Persian | 5,120 | 61% | 234 |
| Arabic | 1,749 | 21% | 80 |
| mixed script | 775 | 9% | 36 |
| Latin | 431 | 5% | 25 (floor) |
| Cyrillic | 256 | 3% | 25 (floor) |
| Hebrew | 3 | <1% | 0 (too few) |
| **Total** | | | **400** |

**This table is provisional.** It rests on the Telegram-only snapshot; once `telegram.parquet` and `twitter.parquet` land, source becomes a second stratum dimension (so that X posts cannot be crowded out by the far larger Telegram volume) and the language shares are recomputed. The total stays ~400.

**Not stratified by channel:** 4,076 of 8,469 posts carry no channel signature, so channel strata would only cover half the data. Instead, **no single channel may contribute more than 5% (20 posts)**; the overflow is redrawn from the same stratum. The largest channel holds 182 posts (2.1%), so this cap should rarely bind.

Within each stratum: simple random sample without replacement, `random_state = 42`.

## 4. Reproducibility

The sample is published as row ids plus the pinned revision hash, so anyone can rebuild the exact set:

```python
pool = df[(df.n_words >= 5) & (df.text != "") & ~df.text.duplicated() & (df.lang_guess != "no_letters")]
alloc = {"fa": 234, "ar": 80, "mixed": 36, "latin": 25, "cyrillic": 25}
sample = pd.concat([
    pool[pool.lang_guess == k].sample(n=min(v, (pool.lang_guess == k).sum()), random_state=42)
    for k, v in alloc.items()
])
```

Versioned per #10. The sample itself stays in the private dataset repo — never in this public one.

## 5. Inter-annotator agreement

#5 requires the labeling methodology to be documented, including agreement if more than one person labels.

**60 posts (15% of the sample)** are labeled independently by a second annotator, drawn proportionally across strata. Reported as Cohen's κ for sentiment and micro-F1 of entity spans for NER (exact boundary and type match). κ below 0.6 means the guidelines are ambiguous and get rewritten before the remaining posts are labeled.

With one annotator, that section states so explicitly, and every model number inherits that caveat.

## 6. Limitations

1. **The window is ~19.5 hours** (2026-08-14 10:31 → 08-15 06:01 UTC). The ±5% margin holds for *this* window only, not for news in general. A model selected here may drift on other days or on breaking-news spikes.
2. **Telegram only.** 5 of 8,469 posts link to X, so the sample cannot support any claim about X content, which #6 asks for.
3. **Language strata come from a script heuristic**, not a language classifier. Persian/Arabic separation relies on script-specific letters and is approximate for short posts.
4. **p = 0.5** is deliberately conservative. Once a first accuracy figure exists, the same formula gives a smaller n for the same precision.

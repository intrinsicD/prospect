# BH-001 coverage-return reanalysis — 2026-07-13

## Source

- Committed input: `bench/hard/results/BH-001-report.json`
- Input SHA-256:
  `a43972e2d5d2aca1b2e514b1bbf38282f61c00c2ebb8c8c9fac11a899f1e8353`
- JSON object: `curiosity`
- Rows: the three paired seeds in the random and curious collection arms

## Extracted observations

| arm | seed | downstream MBRL return | max reward reached | goal-region fraction |
|---|---:|---:|---:|---:|
| random | 0 | 7.567561 | 0.305458 | 0 |
| random | 1 | 5.498681 | 0.236978 | 0 |
| random | 2 | 6.374788 | 0.140115 | 0 |
| curious | 0 | 2.145488 | 0.797367 | 0.019043 |
| curious | 1 | 0.815705 | 0.702037 | 0.002197 |
| curious | 2 | 4.797094 | 0.536751 | 0.002930 |

## Computation and result

- Pearson correlation between maximum reached reward and downstream return over all
  six rows: `-0.864032`.
- Exact two-sided permutation test for Pearson correlation: `21 / 720 = 0.029167`.
  The test exhaustively assigned the six distinct returns to the six fixed proxy values
  and counted absolute correlations at least as large as the observed statistic.
- Spearman correlation between maximum reached reward and downstream return:
  `-0.771429`.
- Exact two-sided rank-permutation test for that statistic:
  `74 / 720 = 0.102778`.
- Spearman correlation between goal-region fraction and downstream return:
  `-0.758971`.
- Random-arm medians (return, maximum reward, goal fraction):
  `(6.374788, 0.236978, 0)`.
- Curious-arm medians: `(2.145488, 0.702037, 0.002930)`.
- Curiosity increased maximum reached reward and reduced return in all three paired
  seeds. The exact two-sided three-pair sign permutation is `2 / 8 = 0.25`.

## Evidential boundary

This is hypothesis-generating evidence, not a causal result. Collection arm jointly
changes the coverage proxy and return; the pooled sample has only six rows; the
within-arm samples have three rows each; and goal fraction contains three tied zeros.
The result shows that these point-coverage proxies are insufficient to explain
downstream control in this committed report. It does not show that reaching high-reward
states causes poor control or that any proposed transition-support mechanism is true.

## Linked research artifacts

- `docs/research/2026-07-13-transformational-research-prompt.md`
- `docs/research/2026-07-13-predictive-reliability-portfolio.md`, Section 2.3


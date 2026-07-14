# OL-002 simulator-oracle localization result

**Status:** completed_localization  
**Scope:** non-gated BridgeControl research evidence

## Outcome

Classification: **mixed:transition_stack,reward_stack**.

The result passed BC-001 replay, mean-wrapper parity, and exact-wrapper parity. Interpretation remains limited to this authored fixture.

## Endpoint and executed-rung results

| Rung | Mean return | Success |
|---|---:|---:|
| `learned_tsinf_penalty` | -2.770126 | 6.25% |
| `learned_tsinf_no_penalty` | -2.660520 | 6.25% |
| `learned_mean_no_penalty` | -4.702069 | 6.25% |
| `exact_target_learned_reward` | 3.213480 | 84.38% |
| `exact_online_learned_reward` | 3.438678 | 84.38% |
| `exact_online_oracle_reward` | 5.273024 | 100.00% |
| `exact_raw` | 5.273024 | 100.00% |
| `prefix_1_target_no_penalty` | -0.428830 | 65.62% |
| `prefix_2_target_no_penalty` | 0.088528 | 53.12% |
| `prefix_4_target_no_penalty` | 1.336343 | 56.25% |
| `prefix_8_target_no_penalty` | 3.333485 | 96.88% |

## Frozen contrasts

| Contrast | Positive seeds | Mean return delta | Gap closed | Regret delta | Decision |
|---|---:|---:|---:|---:|---|
| `penalty_removal` | 6/8 | 0.109606 | 1.36% | 0.000000 | not material |
| `mean_vs_tsinf` | 2/8 | -2.041550 | -25.73% | 0.000000 | not material |
| `transition_stack` | 8/8 | 7.915550 | 79.35% | 0.000914 | material |
| `online_target_interface` | 6/8 | 0.225198 | 10.93% | 0.000000 | not material |
| `reward_stack` | 8/8 | 1.834346 | 100.00% | 0.003293 | material |

## Sequential decision

- Material components: ['transition_stack', 'reward_stack']
- Dominant components: ['transition_stack', 'reward_stack']
- Prefix depths executed: [0, 1, 2, 4, 8, 12]
- Minimum recovery depth: None
- Not run: ['privileged-sequence candidate injection', 'enlarged learned-landscape search', 'MuJoCo P3 replication', 'production or task activation']

## Causal boundary

A→B is only a penalty-coefficient test. B→C bundles member propagation with nonlinear reward averaging. C→D is the transition-mean/recursive-refresh stack. D→E is the online/EMA interface, and E→F is the learned reward composed with online encoding—not reward-head weights alone. No learned-dynamics/oracle-reward decoder arm was fabricated.

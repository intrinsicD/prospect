# Research portfolio: predictive coverage that becomes control

**Repository/domain:** Prospect predictive-world-model agent

**Research question:** When does prediction-error-driven exploration or observation
coverage become reliable downstream control, without replacing Prospect's predictive
spine, distributional Prediction contract, epistemic/aleatoric split, or
one-signal-many-jobs design?

**Literature cutoff:** 2026-07-13

**Sources searched:** live repository code, tasks, ADRs, committed gate and hard-benchmark
artifacts; arXiv; OpenReview/proceedings pages; original project pages; official SLSA
specification. Reviews were used only to expand terms. The novelty labels below are
provisional under this search procedure.

**Result in one sentence:** BC-001 attempted to causally separate node coverage,
bridge connectivity, action excitation, and local density, but its balanced learned
positive control failed; no structural effect was estimated, T1 is retired as the
active mechanism program, and the next evidence branch is simulator-oracle
localization of transition/representation/uncertainty rollout versus reward and search.

## 1. Frontier map

### 1.1 What the repository has actually established

Prospect's stable core is a latent predictive model whose output is a diagonal-Gaussian
Prediction with total variance and scalar epistemic/aleatoric summaries
(src/prospect/types.py:87-109). The same epistemic quantity drives curiosity, mastery,
skill trust, replanning, forgetting, and retrieval; exploit planning instead subtracts
it at every imagined step and can truncate a trajectory after an accumulated bound
(docs/architecture.md:20-55; src/prospect/planning.py:174-218). This is an invariant,
not a placeholder for a second bespoke confidence system.

The shipped upgrade track already includes per-member TS-infinity rollouts, iCEM,
adaptive conformal thresholds, reservoir replay, and distance-kernel top-three
retrieval (tasks/BACKLOG.md:199-221). The authored gates report strong narrow
capabilities: for example, the P1 one-step model beats persistence and ridge, P2 MPC
beats matched CEM-ES on authored Pendulum, P5 hierarchy beats a compute-matched flat
planner, and P8 gated retrieval improves prediction. These are archived results, not
fresh runs in this research turn.

The honest foreign-environment probe is mixed. At 4,096 steps, DMC cartpole balance
scores 92.4 for MBRL versus 99.5 for the matched model-free method, and swing-up scores
6.4 versus 8.8; the authored P2 headline therefore does not reproduce as a decisive
win (bench/hard/results/BH-001-report.md:7-28). Curiosity raises maximum swing-up
reward reached from 0.24 to 0.70 and nonzero goal dwell, but lowers downstream MBRL
return from 6.4 to 2.1 (same report:30-40). Observation imitation succeeds at the same
budget: inverse-dynamics imitation scores 45.3 versus 6.4 from scratch and 0.1 for the
shuffled control; at 512 action labels, watch-then-ground scores 46.2 versus 33.2 for
direct inverse dynamics (same report:42-63). Action-recovery R-squared alone did not
rank reproduced behavior correctly.

Whole-system retrieval supplies another warning about proxy metrics. One-step
retrieval improved prediction but initially damaged planning when distant facts
overrode rollout dynamics; distance coverage and blending made it safe, but its latest
P9 marginal is only +1.4 and the exploit uncertainty penalty is -0.1
(docs/architecture.md:123-129; committed P9 result). A locally useful prediction is
therefore not automatically a useful multi-step intervention.

### 1.2 What is already reserved and must not be renamed as research novelty

Ready U-006 through U-012 cover multi-step loss, latent Mahalanobis OOD, a
latent-versus-ground-truth epistemic audit, latent-action endpoint-swap leakage,
grounding during latent-action training, hierarchy penalty/termination correctness,
and documentation repair (tasks/BACKLOG.md:223-230).

Deferred U-101 through U-112 already reserve terminal value, representation-stack
simplification, epistemic replay prioritization, scalable option search, last-layer
Laplace epistemic, CUSUM termination, continual backprop, semantic consolidation,
predict-then-invert imitation, skill discovery, jumpy consistency, and latent
migration. Each has a trigger and remains deferred until evidence fires it
(tasks/BACKLOG.md:232-244).

### 1.3 Missing evidence and contract mismatches

The main scientific gap is a causal measurement connecting evidence geometry to
control. Current coverage proxies include maximum reached reward, goal-region dwell,
state histograms, nearest-neighbor distance, and aggregate epistemic or error. None
tests whether the data contain a connected directed chain of well-supported
state-action transitions, whether locally feasible actions have distinguishable
effects, or whether the goal region can be stabilized and exited.

The gate grammar also has an attribution gap. Sentinels apply by phase number
(bench/gates.py:332-353), while P12-P14 train separate Pendulum integrity models from
the artifacts earning their capability metrics. Thus “capability plus healthy
sentinels” need not refer to the same model/data/config artifact. Reports omit code,
data, dependency, and model fingerprints, and latest committed run IDs are null. This
is a measurement-research opening, not evidence that the committed metrics are false.

Several implementation audits are important but are not research candidates:
Prediction carries deterministic reward despite architecture prose describing a reward
distribution; Agent observation and target-encoder paths differ; replay omits raw
modality/provenance; and P9's composed loop does not exercise every hierarchy/skill
path. Those belong in bounded correctness tasks if selected, not in a novelty portfolio.

### 1.4 Architectural invariants for all candidates

- Preserve Prediction and its epistemic/aleatoric semantics.
- Keep one-step and jumpy predictive world models as the planning spine.
- Use distance only as support geometry; do not silently replace epistemic uncertainty.
- Keep production interfaces dependency-light and put task-specific fixtures in bench.
- Do not promote a BH result into a gate without changing ADR-0011 deliberately.
- Any selected mechanism later needs a task, an ADR if semantics change, native
  baselines, collapse sentinels, and the full regression ratchet.

## 2. Functional problem signature

### 2.1 Domain-neutral signature

An agent collects finite action-conditioned observations, fits a probabilistic dynamics
model, and optimizes actions through that model. Exploration is rewarded by reducible
prediction uncertainty, while exploitation avoids the same uncertainty. The unresolved
conversion is:

**visited observations → learnable local interventions → supported multi-step paths →
stable action ranking → realized control.**

The failure signature is asymmetric. A collector can visit a high-value state without
learning how to enter it repeatedly, stabilize it, leave it, or predict counterfactual
actions along the route. Global error can fall while a single transition bottleneck
remains; scalar epistemic can be large for either useful missing evidence or irrelevant
novelty; an aggregate capability metric can pass while its integrity evidence belongs
to a different artifact.

### 2.2 Assumptions placed under surgery

1. A visited state is a covered state.
2. More state coverage monotonically improves a controller.
3. Broad state coverage identifies action effects.
4. Coverage is additive and summarized by an average or total.
5. Coverage is a static property of a final dataset.
6. Prediction errors with equal norm matter equally to the planner.
7. A scalar uncertainty magnitude identifies the cause of unreliability.
8. A phase-level sentinel result is evidence about the artifact earning the phase.

### 2.3 Reproducible computation on archived evidence

The reusable prompt required a direct computation before candidate selection. I read
bench/hard/results/BH-001-report.json and paired the three random and three curious
arm-seed rows:

| arm | seed | downstream MBRL return | max reward reached | goal-region fraction |
|---|---:|---:|---:|---:|
| random | 0 | 7.567561 | 0.305458 | 0 |
| random | 1 | 5.498681 | 0.236978 | 0 |
| random | 2 | 6.374788 | 0.140115 | 0 |
| curious | 0 | 2.145488 | 0.797367 | 0.019043 |
| curious | 1 | 0.815705 | 0.702037 | 0.002197 |
| curious | 2 | 4.797094 | 0.536751 | 0.002930 |

Across all six rows, Pearson(max reward, return) is -0.8640 and
Spearman(max reward, return) is -0.7714. Exhaustively re-pairing the six distinct
returns gives two-sided permutation p=0.0292 for Pearson and p=0.1028 for Spearman
(720 permutations). Spearman(goal fraction, return) is -0.7590. Median random versus
curious values are 6.3748 versus 2.1455 for return, 0.2370 versus 0.7020 for maximum
reward, and 0 versus 0.00293 for goal fraction.

This is hypothesis-generating only. The collection arm causes both the proxy and the
return, n=6 is tiny, goal fraction has three ties at zero, and within-arm estimates
have n=3. A paired direction test is correspondingly weak: curiosity increases maximum
reward and reduces return in all three seed pairs, but the two-sided three-pair sign
permutation is p=0.25. The computation therefore does not show that reward coverage
causes poor control. It does establish a concrete anomaly for falsification: under the
current collector, a point-coverage proxy is at best insufficient and can look
anti-predictive when arms are pooled.

The resulting evidence-dependent hypothesis is that isolated point visits are being
counted as coverage while the useful object is a connected, action-conditioned,
locally reliable control support. Its generation provenance is N4-T because this
computation induced the transfer, but the later conceptual audit downgrades the narrow
surviving formulation to N2-T; its mechanism remains unestablished.

## 3. Fixation anti-library

The following directions were rejected before final scoring:

- More data, more capacity, or a larger model: N0 without a location-specific mechanism;
  the report already says three times the exploration budget remained worse.
- Another curiosity reward or a reward/epistemic coefficient sweep: the shipped
  collector already optimizes reward plus epistemic novelty; a weight sweep cannot
  explain why visits are unusable.
- Plain multi-step loss, latent OOD, global epistemic correlation, or option consistency:
  duplicates U-006, U-007/U-008, or U-111.
- Epistemic-prioritized replay, terminal value, Laplace uncertainty, CUSUM, larger option
  search, skill discovery, or PIDM: duplicates U-101/U-103-U-110.
- Retrieve frontier transitions during planning: P9/U-005 already test distance-gated
  blended retrieval and do not answer what evidence geometry makes control possible.
- Generic reachability, safe RL, adaptive horizons, or conformal thresholds: known
  families unless a distinct coverage definition and prediction survive subtraction.
- A coherent-prediction loss by itself: ordinary cross-timescale consistency and close
  to U-111; it survives below only as a diagnostic proposal, not a training upgrade.
- Error-correcting-code majority vote: ordinary ensembling with decorative terminology.
- “Proof attached to Prediction”: breaks the narrow contract and confuses probabilistic
  evidence with deductive proof. Any certificate must be a sidecar.
- Fixing deterministic reward, target-encoder mismatch, replay provenance, or incomplete
  composition: legitimate audits, but N0 correctness work rather than research ideas.

## 4. Productive recombinations

These candidates remain inside a familiar exploration/MBRL grammar. They are useful
because they expose discriminating measurements, not because they are claimed as major
conceptual inventions.

### Candidate R1 — Plan-demanded support holes

**Central claim:** Exploration improves downstream control when it resolves the first
high-epistemic, out-of-support transition on a prospective high-return exploit rollout,
even if it produces less global novelty than ordinary curiosity.

**Novelty class:** N2, a potentially new relationship inside goal-directed active model
learning.

**Known foundation:** Prospect's curiosity collector, iCEM candidate rollouts, P9
distance radius, and localized active learning of dynamics. Capone et al. already target
model accuracy inside a control-relevant region rather than globally
([primary paper](https://openreview.net/forum?id=j6mc0t9Ltsj)).

**Irreducible delta:** The current exploit planner emits a concrete evidence demand
before collection; exploration satisfies that demand rather than maximizing a generic
uncertainty supply.

**Why this is not merely A + B:** A simple goal-conditioned curiosity bonus does not
identify which unsupported transition invalidates the current plan. The distinct claim
is that first-hole satisfaction, not total epistemic reduction, mediates control.

**Changed grammar or transfer mechanism:** The causal direction reverses from
explore-then-plan to plan-question-then-explore, while Prediction remains the only
uncertainty source.

**New prediction:** At equal interaction budget, occupancy-weighted support distance
and error will fall more under first-hole collection than under global curiosity, and
that reduction will predict return after controlling for maximum reward and state
entropy.

**Cheapest killing test:** On the authored swing-up harness, compare random, shipped
curiosity, and first-hole collection at identical 4,096-step/update/planner budgets.
The null hypothesis is no return or exploit-path-error improvement beyond seed noise.
Kill the candidate if it fails to reduce exploit-path error, or if it reduces that
error without improving action ranking or return across five seeds.

**Prior-art threats:** Goal-directed exploration, Bayesian experimental design,
localized GP dynamics learning, MaxInfoRL, and policy-aware model metrics. The current
search makes mechanism novelty unlikely.

**Novelty confidence:** 20-40% that this exact Prospect relationship is unreported,
cutoff 2026-07-13; low confidence because “region of interest” active learning is
mature.

**Scientific value and outcomes:** Success would connect collection location to
planner demand. Partial success would yield a control-relevant error metric. Informative
failure would show that data placement is not the bottleneck and direct work toward
search or model bias.

### Candidate R2 — Decision-flip epistemic curiosity

**Central claim:** Epistemic uncertainty matters for exploration only where ensemble
uncertainty can change the planner's ordering of feasible actions.

**Novelty class:** N2.

**Known foundation:** Ensemble disagreement, iCEM candidate sets, value-aware model
learning, and expected signal-to-noise ratio. VaGraM weights model errors by value
gradients ([primary paper](https://arxiv.org/abs/2204.01464)); the 2026
[policy-aware world-model project](https://policy-aware.github.io/paper-anon/) reports
that ESNR tracks downstream extraction performance.

**Irreducible delta:** A decision-flip probability is computed over the exact candidate
set faced by Prospect's planner; uncertainty common to every candidate is classified as
decision-irrelevant without changing its epistemic meaning.

**Why this is not merely A + B:** Reward-weighted MSE or an epistemic coefficient does
not test whether the chosen action changes. The candidate survives only as an action
ordering diagnostic and acquisition condition.

**Changed grammar or transfer mechanism:** The unit of relevance becomes a comparison
between actions rather than a state or prediction magnitude.

**New prediction:** Member-wise top-action disagreement will predict short-horizon
chosen-action regret better than raw epistemic, and high-disagreement collection will
improve return without necessarily increasing state coverage.

**Cheapest killing test:** At states from regenerated random/curious pools, score a
fixed action set under every ensemble member, compute top-action disagreement, and use
short simulator rollouts to estimate action regret. Null hypothesis: its rank
correlation/AUROC is no better than raw epistemic under a paired bootstrap. Do not build
a collector unless that diagnostic survives all seeds.

**Prior-art threats:** Value-aware models, information-directed control, ESNR, policy
ranking diagnostics, and decision-aware uncertainty. The formulation is probably a
local recombination, not a new primitive.

**Novelty confidence:** 15-35%, cutoff 2026-07-13.

**Scientific value and outcomes:** A positive result gives Prospect a cheap
decision-relevance measurement; a negative result rules out planner-boundary
uncertainty before an expensive collector change.

### Candidate R3 — Frontier latch-and-learn

**Central claim:** Curiosity reaches a valuable frontier but leaves before acquiring a
local transition neighborhood; temporarily holding that region until epistemic
learning progress flattens converts isolated visits into usable dynamics.

**Novelty class:** N1/N2.

**Known foundation:** SurpriseCompetenceMonitor, learning-progress EMAs, curiosity,
region-conditioned curricula, and autonomous practice. DBAP uses a reachability graph
to choose behaviors for repeated autonomous practice
([original project](https://dbap-rl.github.io/)).

**Irreducible delta:** A reached frontier becomes a temporary competence obligation,
with an explicit irrelevant-frontier negative control, rather than just another
novelty reward.

**Why this is not merely A + B:** A new hysteresis threshold is N0. The only scientific
claim is region-specific consolidation until measured local learnability saturates.

**Changed grammar or transfer mechanism:** Coverage acquisition becomes a two-stage
reach-then-consolidate episode.

**New prediction:** Useful-frontier latching increases local transition density,
reduces local epistemic, and improves control while leaving maximum reached reward
roughly unchanged; latching an equally rare reward-irrelevant region does not.

**Cheapest killing test:** Add useful-region and matched irrelevant-region latch arms to
bench/hard/curiosity.py. Null hypothesis: their regional error and final return are
equal. Kill if the useful arm cannot improve local prediction or if the irrelevant arm
helps equally.

**Prior-art threats:** Learning-progress curricula, reset-free autonomous practice,
goal-conditioned rehearsal, and competence progress. It may be an implementation-level
schedule.

**Novelty confidence:** 10-25%, cutoff 2026-07-13.

**Scientific value and outcomes:** Even failure separates sparse dwell from corridor
support and produces reusable regional metrics.

### Candidate R4 — Option-boundary epistemic collection

**Central claim:** In long-horizon tasks, coverage transfers to hierarchical control
only when landing and duration distributions of the options actually selected by the
manager are predictable.

**Novelty class:** N1/N2.

**Known foundation:** JumpyOptionModel, competence-gated options, option-level
Prediction, temporal abstraction, and U-111's reserved consistency loss.

**Irreducible delta:** Acquisition is ranked by existing single-option landing
uncertainty rather than primitive one-step uncertainty; it does not add an option
consistency loss or new skill discovery.

**Why this is not merely A + B:** It is only research-worthy if primitive and option
acquisition rankings diverge and that divergence predicts hierarchical return.

**Changed grammar or transfer mechanism:** The evidence unit moves from primitive
transitions to the boundary distribution consumed by the manager.

**New prediction:** At matched primitive MSE and state coverage, option-directed data
lowers held-out landing/duration error and improves P5 return. The gain disappears when
primitive and option epistemic induce the same ranking.

**Cheapest killing test:** Compare random-option, primitive-curiosity, and
landing-curiosity acquisition on the existing P5 fixture. Null hypothesis: rankings
and returns do not differ. Kill if rankings are nearly identical or lower landing error
does not improve hierarchical action choice.

**Prior-art threats:** Skill-graph exploration, temporal abstraction, and U-111. Any
drift into skill discovery or cross-timescale training is a duplicate.

**Novelty confidence:** 10-25%, cutoff 2026-07-13.

**Scientific value and outcomes:** Its main value is a timescale diagnostic; a null
result would justify keeping primitive curiosity.

## 5. Exploratory candidates

### Candidate E1 — Interventional-rank coverage

**Central claim:** State visitation is insufficient when collected actions do not
identify the locally controllable directions; minimum action-effect rank along exploit
occupancy predicts control.

**Novelty class:** N2-T from system identification and optimal experimental design.

**Known foundation:** Persistent excitation, Fisher-information design, local
controllability, and Prospect's inverse-dynamics auxiliary. Recent AISTATS work derives
input-dependent limits and a persistency-of-excitation condition for active
identification ([primary paper](https://openreview.net/forum?id=zlSHHzj3AU)).

**Irreducible delta:** Coverage is measured over intervention contrasts, not states:
for nearby states, feasible actions must induce distinguishable next-outcome
distributions in every control-relevant direction.

**Why this is not merely A + B:** Generic D-optimal exploration is known. The surviving
question is whether action-effect rank explains Prospect's coverage/control anomaly
after state histograms are exactly matched.

**Changed grammar or transfer mechanism:** State coverage becomes local
interventional-rank coverage, with aleatoric variance defining observation noise and
ensemble epistemic defining model uncertainty.

**New prediction:** With identical state-bin counts, one-sided or correlated actions
produce worse counterfactual-action error and MPC return; local minimum singular value
predicts the gap beyond action entropy.

**Cheapest killing test:** Use PointMass set_state to construct matched-state,
restricted-direction and direction-spanning datasets. Null hypothesis: unseen-action
error and return are equal. Kill if conditioning does not predict counterfactual error
or if improved error does not affect control.

**Prior-art threats:** Classical persistent excitation, active MPC, dual control,
MaxInfoRL, and latent controllability work. Donor-method novelty is zero.

**Novelty confidence:** 25-45% for the recipient relationship, cutoff 2026-07-13.

**Scientific value and outcomes:** A positive result supplies one missing factor in the
recommended factorial fixture; a null result removes action excitation from the
coverage definition.

### Candidate E2 — Occupancy-conditional validity field

**Central claim:** Global calibration can pass while the small state-action-depth
corridor actually used by the planner is systematically under-covered.

**Novelty class:** N2.

**Known foundation:** Adaptive conformal inference, Mondrian/local conformal methods,
trajectory-level calibration, and Prospect's distinct one-step versus CEM thresholds.
UNISafe calibrates trajectory-level epistemic thresholds and emphasizes
latent-action transition uncertainty
([primary paper](https://arxiv.org/html/2505.00779v1)).

**Irreducible delta:** Reliability is an empirical conditional exceedance field indexed
by planner occupancy, action direction, and rollout depth, not another uncertainty
score or global threshold.

**Why this is not merely A + B:** Local conformal calibration is known. The candidate
survives only if conditional failures predict control after global ACI, MSE, and
epistemic/error correlation have passed.

**Changed grammar or transfer mechanism:** A global “covered/uncovered” claim becomes a
use-conditioned local validity statement.

**New prediction:** One corridor exhibits reproducible conditional undercoverage before
control fails; its exceedance rate predicts return better than global calibration.

**Cheapest killing test:** Re-bin existing nominal and rollout audit scores by
ground-truth phase, action sign, and depth, with bins fixed on a pilot split. Null
hypothesis: independent-audit exceedance is exchangeable across bins after multiplicity
control. Kill if no stable undercoverage appears.

**Prior-art threats:** Local conformal prediction, risk-controlling prediction,
UNISafe, and U-003/U-008. This is a measurement study, not a threshold upgrade.

**Novelty confidence:** 10-30%, cutoff 2026-07-13.

**Scientific value and outcomes:** Failure is useful: it would show that calibration
locality is not the cause of the BH anomaly.

### Candidate E3 — Dynamic competence stock

**Central claim:** Coverage is not monotone: local predictive competence can be acquired
and later depleted by updates elsewhere, and control depends on the minimum current
stock along its occupancy flow.

**Novelty class:** N3 provisional; likely downgrade to N2 after a continual-learning
audit.

**Known foundation:** Forgetting detection, local learning progress, rehearsal,
plasticity loss, and nonstationary competence models.

**Irreducible delta:** “Coverage” becomes a time-indexed balance variable with measured
acquisition and depletion, rather than membership in a growing visited set.

**Why this is not merely A + B:** A recency-weighted count is not enough. The distinct
claim requires novelty training to increase global coverage while depleting a
task-critical local stock that predicts control.

**Changed grammar or transfer mechanism:** The state of the research object includes
evidence history and interference, so identical final visitation counts can have
different current competence.

**New prediction:** Alternating exploit-corridor and novelty-heavy updates yields a
return drop preceded by local error/epistemic stock depletion; localized rehearsal
restores return without restoring global coverage metrics.

**Cheapest killing test:** Train on one fixed multiset under block, reverse-block,
alternating, and globally shuffled schedules while hashing identical data exposure.
Null hypothesis: final regional predictions and return are order-invariant. Kill or
downgrade if deterministic minibatches remove the effect.

**Prior-art threats:** Continual learning, catastrophic forgetting, hysteresis,
curriculum order, and U-103/U-107. An ordinary order effect is N1/N2.

**Novelty confidence:** 10-25%, cutoff 2026-07-13.

**Scientific value and outcomes:** A positive result would show coverage is a
model-history property; a null result cleanly directs research back to evidence
geometry.

### Candidate E4 — Projective prediction-family audit

**Central claim:** Flat, multi-step, jumpy-option, inverse, and retrieved predictions
about the same outcome should agree after composition or marginalization; the pattern
of disagreement may diagnose a failure hidden by each marginal metric.

**Novelty class:** N2-T after adversarial downgrade.

**Known foundation:** Forecast reconciliation maps incoherent forecasts to a coherent
constraint set ([primary paper](https://arxiv.org/abs/2103.11128)), cycle consistency,
multiscale world models, and U-111's cross-timescale consistency.

**Irreducible delta:** Initially only an audit graph of shared-outcome constraints; no
new loss is proposed. The potentially useful residue is locating which abstraction
edge breaks rather than averaging inconsistency.

**Why this is not merely A + B:** As a loss, it is A+B and duplicates U-111. It remains
only as a pre-training diagnostic that can reject the assumption that one-step and
option models fail in the same place.

**Changed grammar or transfer mechanism:** Predictions are evaluated as a family linked
by explicit pushforward/composition relations, not independently.

**New prediction:** Two checkpoints with matched one-step and option errors can have
different cross-path residual patterns, and only one pattern localizes downstream
control failure.

**Cheapest killing test:** Compare one-step-composed and jumpy predictions on fixed P5
trajectories and inject a localized option bias. Null hypothesis: the residual graph
adds no localization AUROC beyond each model's NLL/epistemic. Kill if so.

**Prior-art threats:** Forecast reconciliation, multiscale state-space models, cycle
consistency, MTS3, MCPlanner, and U-111. OpenReview challenge pages blocked full-text
inspection for two nearest works, increasing audit uncertainty.

**Novelty confidence:** 5-20%, cutoff 2026-07-13.

**Scientific value and outcomes:** Primarily a diagnostic building block for the
trajectory-syndrome candidate; not independently recommended.

## 6. Transformational candidates

Four first-pass formulations changed the primitive or evidence grammar. The
adversarial audit then rejected the broad novelty claims for T1-T3. They remain here to
show the full research path and their smallest testable residues; only T4 remains a
provisional N3 formulation pending a comparably deep audit. A portfolio in which the
audit eliminates most “transformational” language is a stronger result than preserving
inflated labels.

### Transformational candidate T1 — Viable predictive conductance

**Central claim:** A dataset supports downstream control only when it opens a
task-relevant directed flow of action-excited, predictively reliable transitions from
the start basin into a goal basin that can be stabilized or exited; node visitation and
average prediction quality are not sufficient.

**Novelty class:** Generation provenance N4-T because the six-row computation induced
the transfer hypothesis; post-audit conceptual class N2-T only for a narrow correlated
reliability scaling conjecture. The broad “coverage is a reliable graph/min-cut”
formulation is rejected as N1.

**Known foundation:** Reachability and viability theory; probabilistic roadmaps; graph
world models; task/skill graphs; occupancy coverage; and network reliability. L3P
learns latent landmarks and reachability edges
([primary paper](https://arxiv.org/abs/2011.12491)); DBAP builds a directed task graph
and practices toward uniform goal coverage
([original project](https://dbap-rl.github.io/)); the 2025 skill-graph work explicitly
seeks robust, reliable edges and prioritizes reward-relevant practice
([primary paper](https://openreview.net/forum?id=vjT2aL6Wlg)); ROVER maximizes occupancy
coverage with a resolvent world model
([primary preprint](https://arxiv.org/abs/2606.21271)); UNISafe augments latent
reachability with calibrated epistemic uncertainty
([primary paper](https://arxiv.org/html/2505.00779v1)).

**Irreducible delta:** After audit, only this remains: define a directed,
dependence-aware two-terminal reliability or minimal-cut functional from calibrated
world-model evidence and prospectively test a reproducible control-collapse threshold
and finite-size scaling over topology, data budget, and abstraction resolution. A
frozen transition graph whose capacities combine support, action excitation, and
predictive reliability is one estimator, not itself a novel primitive.

One operational prototype is:

- freeze either ground-truth fixture coordinates or an encoder checkpoint;
- partition state space into cells and actions into local directions;
- assign edge capacity as the product of (a) capped effective transition count,
  (b) normalized local action-effect minimum singular value, and (c) a held-out lower
  confidence bound on staying below a predeclared prediction/action-ranking error;
- permit a goal cell as a sink only if short standardized perturbations can be
  corrected or returned from;
- compute source-to-sink max flow and the bottleneck cut.

The exact estimator is not claimed as theory; the causal prediction is about the
non-additive path/cut structure.

**Why this is not merely A + B:** At broad scope, it is A + B: learned reachability
graphs, bottleneck/coverage theory, and classical path/cut reliability are mature and
already adjacent. The only nontrivial residue is a preregistered correlated
control-reliability scaling law that must beat L1 coverage, sequence-level
concentrability, bottleneck centrality, and conformal reachability under matched
ordinary metrics.

**Changed grammar or transfer mechanism:** Coverage changes from a scalar property of
visited points to a task-indexed flow/cut property of reliable interventions. Adding
evidence outside every source-goal cut can have exactly zero marginal value; closing
one critical cut can have a discontinuous effect.

**New prediction:** In an authored factorial fixture, bridge-edge presence and local
action excitation interact to cause a large control-return increase even when every
state cell, total transition count, nuisance coverage, and global one-step MSE are
matched. Adding many non-bridge transitions will not substitute for the missing edge.
Across cells, conductance will rank return better than maximum reward, visitation
entropy, nearest-neighbor distance, mean epistemic, and global MSE.

**Cheapest killing test:** Run the BridgeControl experiment in Section 10: a preregistered
2×2×2 manipulation of bottleneck edge, action rank, and local density with five seeds,
exact-dynamics and balanced-data positive controls, action-label and nuisance-coverage
negative controls. The null hypothesis is that, after matching ordinary metrics,
structural factors add no predictive or causal value. Abandon T1 if return is fully
explained by global MSE or if the bridge factor fails on two bottleneck lengths.

**Prior-art threats:** The broad novelty claim is destroyed. L3P already supplies
learned reachability graphs
([ICML paper](https://proceedings.mlr.press/v139/zhang21x.html)); adaptive-grid
exploration uses confident subgoal edges
([ICML paper](https://openreview.net/forum?id=59MYoLghyk)); coverability formalizes
coverage for arbitrary downstream rewards
([ICML paper](https://proceedings.mlr.press/v235/amortila24a.html)); sequence-level
data-policy coverage controls downstream error amplification
([UAI paper](https://proceedings.mlr.press/v286/zhou25a.html)); learned edge
probabilities already drive explicit path-and-cut search in motion planning
([RSS paper](https://arxiv.org/abs/2305.10395)); and UNISafe combines calibrated
epistemic uncertainty with latent reachability. Classical two-terminal reliability
and probabilistic roadmaps own the donor mathematics. The scoped surviving claim is
only the correlated threshold/scaling prediction.

**Novelty confidence:** 5-15% for the broad formulation and 20-35% that the narrow
correlated scaling delta is irreducible, cutoff 2026-07-13. Correlated, directed,
policy-dependent edge failures and moving abstractions may make “percolation” only a
metaphor.

**Scientific value:** It directly explains why reaching a goal can fail to teach
control and unifies data placement, model reliability, and stabilizability without
adding another uncertainty signal.

**Publishable if successful:** A causal coverage-to-control law, estimator, authored
fixture, and foreign-environment replication.

**Publishable if partially successful:** A decomposition showing that bridge support,
action excitation, or local density—not the full composite—is decisive.

**Publishable if it fails informatively:** A strong counterexample showing that global
MSE or planner search, rather than evidence topology, explains the anomaly.

### Transformational candidate T2 — Continuous trajectory syndrome

**Central claim:** The vector pattern of uncertainty-normalized consistency residuals
across redundant predictive routes can identify where and why a world-model trajectory
is unreliable even when total NLL and scalar epistemic magnitude are matched.

**Novelty class:** N1 after audit. A formal Prospect-head diagnosability matrix may earn
narrow N2 if it proves unique/simultaneous-fault isolability and improves repair
selection. “New trajectory syndrome theory” is rejected.

**Known foundation:** Model-based fault diagnosis designs structured residuals that
isolate unknown faults, including neural residuals trained only on healthy data
([primary paper](https://arxiv.org/abs/1910.05626)); sensor-wise residuals can be more
interpretable than aggregate residuals
([primary comparison](https://arxiv.org/abs/2309.02274)). WFM-Eval similarly decomposes
video-world-model failures instead of trusting holistic scores
([primary paper](https://openreview.net/forum?id=gvtwynIibB)). Prospect already has
flat, composed, jumpy, inverse, and retrieved predictive paths plus U-111's reserved
consistency loss.

**Irreducible delta:** A narrow application-specific delta remains: define a formal
diagnosability matrix whose rows are Prospect-native one-step, long-horizon/TS-infinity,
jumpy-option, inverse, and retrieval paths and whose columns are explicitly injected
failure modes. Whiten residuals by heteroskedastic aleatoric variance, reserve epistemic
for support violations, derive isolability conditions, and use the recovered signature
to select a repair action.

**Why this is not merely A + B:** At broad scope it is A + B and old terminology:
parity-space analytical redundancy, nonlinear residual design, learned residual banks,
and one-/multi-step control-aware surrogate residuals already exist. The candidate
survives only if the exact head matrix yields a new isolability result and better
intervention choice.

**Changed grammar or transfer mechanism:** A prediction failure is no longer observed
as one magnitude. It is an equivalence class of residual patterns under a typed family
of predictive checks, with aleatoric covariance specifying tolerance and epistemic
disagreement specifying reducibility.

**New prediction:** For failures calibrated to have the same total NLL and epistemic
mean, syndrome structure identifies the corrupted segment/type above 0.8 macro-AUROC
and predicts which mitigation—collect, abstain, reject retrieval, or audit
representation—restores control. Aggregate baselines remain near chance between at
least two matched failure types.

**Cheapest killing test:** On P5/P9 fixtures, inject one localized transition bias, one
action-label corruption, and one far-fact retrieval corruption at matched error energy.
Compare a preregistered linear syndrome decoder with total surprise, epistemic,
per-coordinate residuals, and a generic MLP on raw residuals. Null hypothesis:
structured checks add no held-out fault-localization or control-loss prediction. Use
semantics-preserving latent rotations as a negative control.

**Prior-art threats:** Classical analytical redundancy already defines parity spaces,
fault signatures, detectability, and isolability for nonlinear dynamics; learned
residual banks and combined one-/multi-step dynamic surrogate residuals bridge the
donor into learned control. Most damagingly, the 2026 world-action-model paper
[Is the Future Compatible?](https://arxiv.org/abs/2605.07514) already uses
action-state/inverse consistency to diagnose imagined-rollout failure. Shared encoders
also make unique localization doubtful.

**Novelty confidence:** 0-10% for the broad theory and 10-25% for an irreducible
Prospect-head isolability result, cutoff 2026-07-13.

**Scientific value:** It could turn one-signal-many-jobs from a scalar alarm system
into an interpretable diagnostic layer while keeping the signal and contract intact.

**Publishable if successful:** A typed failure benchmark plus a diagnostic primitive
that localizes predictive-control faults.

**Publishable if partially successful:** Evidence that only one redundancy relation
is useful, supplying a targeted sentinel.

**Publishable if it fails informatively:** A demonstration that shared-model error
correlation makes analytical redundancy ineffective for learned latent dynamics.

### Transformational candidate T3 — Artifact-bound counterfactual witness graph

**Central claim:** A capability claim is supported only by evidence bound to the exact
artifact/data/config dependency graph and responsive to an intervention that breaks the
claimed mechanism; phase-level healthy sentinels are insufficient evidence attribution.

**Novelty class:** N0-N1 infrastructure after audit. A formal counterfactual revocation
operator with minimal-witness semantics may earn narrow N2; the broad proof/artifact
formulation is rejected.

**Known foundation:** SLSA provenance records where, when, and how an artifact was
produced ([official specification](https://slsa.dev/spec/v1.2/provenance)); AMLAS
structures an evidence base for ML assurance
([primary guidance](https://arxiv.org/abs/2102.01564)); causal assurance explicitly
relates evidence to safety causalities
([primary paper](https://arxiv.org/abs/2201.05451)); mutation testing evaluates whether
tests detect injected faults, including real RL fault operators
([muPRL](https://arxiv.org/abs/2408.15150)).

**Irreducible delta:** Only a counterfactual revocation operator remains: a live claim
predicate pins code, config, data, seeds, environment, positive evidence, and mandatory
negative sentinels; every support edge declares how it must react to a
mechanism-breaking intervention. Drift, corruption, or sentinel failure retracts the
claim, and minimal witness cuts expose claims still apparently supported after a
load-bearing artifact is broken.

**Why this is not merely A + B:** The broad proposal is A + B and nearly directly
anticipated. It survives scientifically only if mandatory counterfactual sensitivity
and automatic revocation catch stale or falsely supported claims that CAF-style
machine-readable assertions, in-toto provenance, AMLAS, and ordinary negative controls
do not.

**Changed grammar or transfer mechanism:** The evidence unit changes from
phase → aggregate metrics to claim → exact artifact → counterfactual witness. Passing
is a property of a connected evidence subgraph, not a bag of current checks.

**New prediction:** Corrupting or replacing the P12/P13 capability artifact while
leaving the separate Pendulum sentinel artifact healthy will leave current
phase-applicable sentinel status unchanged, whereas an artifact-bound witness will
fail and localize the unsupported claim. An explicitly typed environment-health
witness will correctly remain healthy.

**Cheapest killing test:** Without changing gate behavior, serialize one exact
capability-artifact fingerprint and run one representation/uncertainty probe on it.
Then corrupt that artifact only. Null hypothesis: binding adds no detection or
localization beyond the current phase sentinel. Negative control: change an unrelated
artifact and require the witness to remain stable.

**Prior-art threats:** The broad claim is destroyed. Proof-carrying code/data and
in-toto bind properties and computation history to artifacts; AMLAS structures ML
claim-argument-evidence. The near-exact 2026 threat,
[Composable Assurance for AI Alignment](https://ojs.aaai.org/index.php/AAAI/article/view/41151),
defines machine-readable properties tied to specific AI artifacts, composes them into a
living DAG, and blocks noncompliant deployment. Negative controls, canaries, and ML
test-score practices cover “sentinel” logic. A verifier also cannot prove empirical
generalization without stating its assumptions.

**Novelty confidence:** 0-5% for the broad formulation and 10-25% for irreducible
counterfactual revocation/minimal-witness semantics, cutoff 2026-07-13. Engineering
value likely exceeds research novelty.

**Scientific value:** It prevents false attribution in every future research gate and
makes negative controls first-class evidence.

**Publishable if successful:** A compact executable evidence grammar demonstrated on
real benchmark attribution failures.

**Publishable if partially successful:** A mutation/witness matrix that exposes which
current claims are underdetermined.

**Publishable if it fails informatively:** Evidence that simple artifact binding plus
existing sentinels is sufficient, downgrading the proposal to a hygiene task.

### Transformational candidate T4 — Epistemic job-compatibility polytope

**Central claim:** Reusing one epistemic ordering across exploration, exploitation,
retrieval, mastery, and forgetting is empirically justified only when a nonempty region
of monotone calibrations and job policies meets all predeclared downstream regret
limits simultaneously.

**Novelty class:** N3 provisional; possibly N3-T if treated as a feasibility-geometry
transfer from robust multiobjective design.

**Known foundation:** Decision calibration evaluates predictions through downstream
decision makers ([primary paper](https://arxiv.org/abs/2107.05719)); 2026
decision-aligned UQ shows that common uncertainty metrics can be misaligned with
utility ([primary paper](https://arxiv.org/abs/2606.26990)). Prospect already uses
separate job thresholds and opposite explore/exploit signs.

**Irreducible delta:** For a frozen epistemic output u, define each job's acceptable
set as the family of monotone normalizations, thresholds, signs fixed by semantics, and
job actions whose regret is below epsilon. The compatibility object is their
intersection; its emptiness, volume, and active constraints state whether the same
epistemic ordering can genuinely serve all jobs.

**Why this is not merely A + B:** Measuring each job separately cannot reveal an
incompatible reuse: every job could pass under a different ordering or model. The
candidate tests simultaneous compatibility of the same frozen signal and artifact.

**Changed grammar or transfer mechanism:** “One signal, many jobs” changes from an
architectural slogan to a falsifiable feasibility claim. A successful component is not
enough; the common acceptable region must be nonempty and stable under a held-out
environment.

**New prediction:** The compatibility region is large on an analytic
linear-Gaussian positive control, narrow on PointMass, and empty or unstable on DMC
swing-up: calibration settings that increase useful exploration will conflict with
exploit/retrieval regret. If the current configuration lies in a stable nonempty
region, the concern is falsified.

**Cheapest killing test:** Freeze one model per seed, cache candidate predictions and
job outcomes, and sweep only monotone normalizations and existing job thresholds
without retraining. Include P3 exploration, P8/P9 retrieval, P9 exploitation, and a
mastery probe. Null hypothesis: a broad region contains the current settings on all
held-out seeds. Use an analytic correctly specified model as positive control and
permuted epistemic order as negative control.

**Prior-art threats:** Decision calibration and decision-aligned UQ are close and may
subsume most of the conceptual contribution; multiobjective feasibility regions,
selective prediction, and calibration transfer are additional threats. The simultaneous
opposite-sign multi-job intersection is the necessary delta.

**Novelty confidence:** 15-35%, cutoff 2026-07-13.

**Scientific value:** It tests Prospect's most distinctive design invariant without
adding a new signal. An empty region would identify which job breaks reuse and whether
job-specific calibration suffices.

**Publishable if successful:** A simultaneous decision-compatibility criterion with
cross-environment evidence.

**Publishable if partially successful:** A map of active conflicts that constrains
which jobs can share epistemic ordering.

**Publishable if it fails informatively:** Strong evidence that Prospect's current
one-signal design is compatible over the tested envelope.

### 6.1 Nearest-work matrix for serious candidates

| Candidate | Nearest primary work | What is already there | Scoped delta that must survive | Audit verdict |
|---|---|---|---|---|
| T1 viable predictive conductance | [L3P](https://proceedings.mlr.press/v139/zhang21x.html), [coverability](https://proceedings.mlr.press/v235/amortila24a.html), [sequence coverage](https://proceedings.mlr.press/v286/zhou25a.html), [path/cut search](https://arxiv.org/abs/2305.10395), [UNISafe](https://arxiv.org/html/2505.00779v1) | learned reachability edges, downstream coverage, error amplification, explicit cut search, calibrated latent reachability | correlated threshold and finite-size scaling that beats all of these | broad novelty rejected; narrow N2-T only |
| T2 continuous trajectory syndrome | [Neural structured residuals](https://arxiv.org/abs/1910.05626), [learned dynamics diagnosis](https://arxiv.org/abs/2305.04670), [world-action consistency](https://arxiv.org/abs/2605.07514) | learned residual banks, nonlinear dynamic fault diagnosis, action/inverse rollout consistency | formal Prospect-head isolability matrix plus repair selection | broad theory rejected; N1 or narrow N2 |
| T3 artifact-bound witness graph | [in-toto](https://www.usenix.org/conference/usenixsecurity19/presentation/torres-arias), [AMLAS](https://arxiv.org/abs/2102.01564), [CAF](https://ojs.aaai.org/index.php/AAAI/article/view/41151) | exact artifact provenance, ML evidence cases, machine-readable artifact assertions and living DAG | comparative counterfactual revocation/minimal-witness operator | broad research claim rejected; infrastructure or narrow N2 |
| T4 job-compatibility polytope | [Decision calibration](https://arxiv.org/abs/2107.05719), [decision-aligned UQ](https://arxiv.org/abs/2606.26990) | evaluate uncertainty by downstream decision utility | simultaneous feasible region for one frozen ordering serving opposite-sign jobs | survives narrowly; direct 2026 threat lowers confidence |
| E1 interventional-rank coverage | [Persistent-excitation active ID](https://openreview.net/forum?id=zlSHHzj3AU) | experiment inputs determine identification sample complexity | causal role under matched state coverage in Prospect | N2-T only |
| R2 decision-flip uncertainty | [VaGraM](https://arxiv.org/abs/2204.01464), [policy-aware WMs](https://policy-aware.github.io/paper-anon/) | value-sensitive model learning and downstream policy metric | top-action disagreement on Prospect's exact candidate set | N2 only |
| E4 projective family | [Forecast reconciliation](https://arxiv.org/abs/2103.11128), U-111 | coherent constrained forecasts and jump consistency | diagnostic localization before any new loss | downgraded to N2-T |

## 7. Cross-domain transfers

The maps below separate a donor mechanism from a superficial metaphor. Each transfer
was checked by removing field terminology, mapping causal roles, naming at least three
broken correspondences, comparing a native baseline, and asking whether the same
proposal would have been obvious before the enabling repository evidence.

### Transfer 1 — Reliability percolation into predictive control support

**Candidate:** T1 viable predictive conductance

**Donor field:** Network reliability and percolation theory

**Recipient field:** Learned-dynamics evidence for model-predictive control

**Domain-neutral functional problem:** Determine whether uncertain local components
jointly provide an end-to-end route between required terminals.

| Structural role | Donor field | Recipient field |
|---|---|---|
| State | network condition | frozen transition-evidence graph |
| Observation | component test | held-out transition prediction and support count |
| Action/operator | open/repair an edge | collect or rehearse a state-action transition |
| Objective/energy | terminal connectivity/flow | reliable start-to-viable-goal control flow |
| Invariant | cut separates terminals | every controller must cross a start-goal cut |
| Noise/uncertainty | component failure probability | epistemic/aleatoric predictive failure |
| Boundary condition | source and sink | reset distribution and stabilizable goal basin |
| Failure mode | one critical cut closes | unsupported bridge reverses action ranking |

**Preserved causal mechanism:** High average component quality does not imply terminal
connectivity. A small cut can dominate global reliability, and path redundancy can
make end-to-end behavior robust.

**Broken correspondences:**

1. Latent regions and edges move during representation learning—correctable by frozen
   checkpoints or ground-truth fixture coordinates.
2. Edge failures share model parameters and are correlated—scientifically productive
   but potentially fatal to naive independent network-reliability formulas.
3. Actions change which graph is reachable—scientifically productive; the graph must be
   directed and policy/task indexed.
4. Continuous dynamics require a partition—correctable only if conclusions survive
   multiple preregistered resolutions.

**Required invention:** A dependence-aware, action-conditioned edge-capacity estimator
and a viability condition for sink nodes; no literal percolation theorem is claimed.

**Recipient-specific prediction:** Closing one bridge cut changes return sharply while
adding equal non-bridge evidence does not, under matched state coverage/global MSE.

**Native competitor:** Maximum reward, goal dwell, state entropy, nearest-neighbor
distance, global MSE, mean/accumulated epistemic, and ordinary reachability.

**Adoption barrier:** Continuous partitions can manufacture bottlenecks, and causal
graph construction from moving latents may cost more than the initial experiment.

**Enabling change:** Prospect already exposes exact-state fixtures, Transition records,
ensemble Predictions, and planner paths; BH supplies the point-coverage anomaly.

**Transfer novelty:** Donor-method novelty: none. Correspondence novelty: low after the
path/cut, coverability, and latent-reachability audit. Recipient-adaptation novelty:
low-medium. Prediction novelty: medium only for correlated finite-size scaling.
Overall: broad N1; narrow N2-T. Its generation provenance remains N4-T because the
repository computation induced the hypothesis.

### Transfer 2 — Analytical redundancy into continuous trajectory syndromes

**Candidate:** T2 continuous trajectory syndrome

**Donor field:** Error-correcting codes and model-based fault diagnosis

**Recipient field:** Learned latent world-model diagnostics

**Domain-neutral functional problem:** Infer the location/type of a hidden fault from a
pattern of violated redundant relations without observing a clean latent truth.

| Structural role | Donor field | Recipient field |
|---|---|---|
| State | codeword/system health | latent trajectory and predictive subsystem state |
| Observation | parity/residual vector | normalized cross-path prediction residuals |
| Action/operator | parity check/residual generator | compose, jump, invert, retrieve, mask |
| Objective/energy | detect and isolate corruption | localize unsupported segment/mechanism |
| Invariant | valid words satisfy checks | equivalent predictive paths agree in distribution |
| Noise/uncertainty | channel/sensor noise | aleatoric covariance |
| Boundary condition | fault basis/code design | available independent prediction routes |
| Failure mode | ambiguous syndrome | shared-model error makes every check co-fail |

**Preserved causal mechanism:** Structured redundancy makes an unobserved local failure
observable; the residual pattern carries more localization information than its norm.

**Broken correspondences:**

1. Predictive routes share encoders and training data, so checks are not independent—
   scientifically productive and potentially fatal.
2. There is no fixed generator matrix or discrete corruption budget—correctable with a
   learned/empirical covariance and a preregistered fault basis.
3. Residuals are continuous and heteroskedastic—correctable using aleatoric
   normalization, but calibration error can masquerade as a syndrome.
4. Some “faults” are legitimate stochasticity rather than defects—handled only if the
   epistemic/aleatoric distinction survives injected-noise controls.

**Required invention:** A typed continuous check family and decoder robust to correlated
residuals and unseen fault combinations.

**Recipient-specific prediction:** At matched total NLL/epistemic, structured syndromes
localize fault type and trajectory segment better than scalar or unstructured baselines.

**Native competitor:** Surprise, scalar epistemic, per-dimension residuals, WFM-style
error taxonomy, and an unrestricted classifier on raw model outputs.

**Adoption barrier:** Constructing genuinely independent checks may require extra
models; without independence the code analogy collapses into ordinary diagnostics.

**Enabling change:** Prospect has flat/jumpy/inverse/retrieval paths over one shared
latent and already records decomposed Prediction and Surprise.

**Transfer novelty:** Donor-method novelty: none. Correspondence novelty: low because
learned dynamics and world-action consistency already bridge the fields. Recipient
adaptation: low-medium. Prediction novelty: low-medium only for formal simultaneous
isolability and repair choice. Overall: N1 or narrow N2.

### Transfer 3 — Proof-carrying artifacts into counterfactual gate evidence

**Candidate:** T3 artifact-bound counterfactual witness graph

**Donor field:** Proof-carrying code, software attestations, runtime verification, and
ML assurance

**Recipient field:** Benchmark-gated agent research

**Domain-neutral functional problem:** Accept a claim from a complex producer only when
a smaller checker can bind supporting evidence to the exact object and assumptions.

| Structural role | Donor field | Recipient field |
|---|---|---|
| State | artifact plus dependency graph | model/data/config/code research artifact |
| Observation | signed attestation/check result | metric, sentinel, intervention response |
| Action/operator | verify proof/attestation | traverse dependencies and execute witness |
| Objective/energy | justified artifact acceptance | justified capability/claim support |
| Invariant | evidence subject matches artifact | witness consumes claimed artifact/dependency |
| Noise/uncertainty | compromised builder/stale input | stochastic run, seed noise, stale model/store |
| Boundary condition | declared trusted checker | declared gate semantics and intervention basis |
| Failure mode | valid proof for wrong subject | healthy sentinel from a different model |

**Preserved causal mechanism:** Trust moves from the producer's complexity to explicit
binding and a small check over stated dependencies.

**Broken correspondences:**

1. Gate evidence is statistical rather than deductive—scientifically productive; the
   result is an assurance witness, not a mathematical proof.
2. Training and data are stochastic—correctable with distributions and seed-aware
   statements, not hashes alone.
3. Model updates invalidate cached evidence—correctable with content/version
   dependencies and freshness.
4. The checker may share code with the producer—potentially fatal unless negative
   controls and mutation operators are sufficiently independent.

**Required invention:** Claim-indexed counterfactual support semantics over an
artifact-dependency hypergraph.

**Recipient-specific prediction:** Current phase sentinels can stay green after the
capability artifact is corrupted; an artifact-bound witness fails and identifies the
unsupported claim while unrelated context witnesses stay healthy.

**Native competitor:** Current GateReport plus run ID, simple hashes/provenance, phase
sentinels, and mutation score.

**Adoption barrier:** The evidence graph can become process-heavy; a useful prototype
must demonstrate localization with one or two claims before any framework expansion.

**Enabling change:** Current P12-P14 artifact separation supplies a concrete falsifiable
case; SLSA and RL mutation testing supply mature donor conventions.

**Transfer novelty:** Donor novelty: none. Correspondence novelty: very low after the
CAF/in-toto/AMLAS audit. Adaptation novelty: low. Prediction novelty: low-medium only
for comparative counterfactual revocation. Overall: N0-N1 infrastructure or narrow N2.

### Transfer 4 — Adjoint error estimation into planner-dual coverage

**Candidate:** Planner-dual error certificate

**Donor field:** Goal-oriented adaptive numerical analysis and adjoint error estimation

**Recipient field:** Model-predictive control with learned latent dynamics

**Domain-neutral functional problem:** Weight local approximation residuals by their
causal influence on a final quantity of interest.

| Structural role | Donor field | Recipient field |
|---|---|---|
| State | numerical solution trajectory | imagined latent rollout |
| Observation | local equation residual | prediction residual/epistemic |
| Action/operator | refine mesh/basis | collect transition or shorten/alter plan |
| Objective/energy | quantity-of-interest error | return/action-ranking error |
| Invariant | dual transports local error | planner sensitivity transports model error |
| Noise/uncertainty | discretization/measurement error | epistemic and aleatoric prediction error |
| Boundary condition | terminal functional | task reward/goal and current planner state |
| Failure mode | small norm, large QoI error | low MSE, high control regret |

**Preserved causal mechanism:** Equal local errors can have unequal downstream effect;
a backward sensitivity exposes their leverage on the output functional.

**Broken correspondences:**

1. iCEM selection is stochastic and nonsmooth—correctable only approximately through
   finite differences or fixed candidate sets.
2. True future residuals are unavailable at plan time—scientifically productive;
   ensemble disagreement is only a proxy.
3. Learned latent coordinates are not physical conserved variables—correctable only if
   the ranking is invariant to benign latent reparameterizations.
4. Planner sensitivity changes after every receding-horizon step—requires local,
   short-lived certificates.

**Required invention:** A robust finite-difference or influence approximation tied to
candidate-ranking change rather than differentiable value alone.

**Recipient-specific prediction:** Matched-norm errors aligned with high dual
sensitivity cause larger action-ranking and return loss; dual-weighted epistemic
outpredicts raw accumulated epistemic.

**Native competitor:** VaGraM/value-aware model error, ESNR, decision-flip epistemic,
global MSE, and rollout error.

**Adoption barrier:** Estimating stable sensitivities through iCEM may multiply planner
compute and still be dominated by candidate-set sampling noise.

**Enabling change:** FlatPlanner exposes a pure imagined-return functional and exact
fixtures can inject controlled latent prediction biases.

**Transfer novelty:** Donor novelty: none. Correspondence: medium. Adaptation: medium.
Prediction: low-medium because value-aware models are close. Overall: N2-T, possibly
N3-T only if a coordinate-invariant control certificate survives.

### Transfer 5 — Viability theory into recoverable coverage

**Candidate:** Probabilistic viable coverage

**Donor field:** Viability theory, controlled invariance, and robust MPC

**Recipient field:** Exploration-data sufficiency for learned control

**Domain-neutral functional problem:** Distinguish entering an acceptable state from
being able to remain within, recover to, or safely leave an acceptable set under
admissible controls and uncertainty.

| Structural role | Donor field | Recipient field |
|---|---|---|
| State | physical system state | Prospect latent or fixture state |
| Observation | known/estimated dynamics | Prediction distribution |
| Action/operator | admissible control | primitive action or option |
| Objective/energy | remain in/recover to constraint set | sustain goal/corridor control under low error |
| Invariant | viability kernel | recursively usable predictive support |
| Noise/uncertainty | bounded disturbance | aleatoric spread plus epistemic miss |
| Boundary condition | constraint/target set | task-acceptable and calibrated region |
| Failure mode | enter state but cannot remain/recover | fly-through goal visit without control |

**Preserved causal mechanism:** Recursive admissibility separates reachability from
sustained controlled use.

**Broken correspondences:**

1. Classical kernels usually assume known dynamics—scientifically productive; Prospect
   has learned stochastic Predictions.
2. Hard disturbance bounds do not map cleanly to calibrated probabilistic tails—
   correctable only with an explicit risk level.
3. Infinite-horizon invariance is unnecessary and infeasible for current tasks—
   correctable with finite-horizon dwell/recovery.
4. Latent geometry may not preserve physical constraints—potentially fatal outside
   authored ground-truth probes.

**Required invention:** A finite-horizon empirical viability predicate over Prediction
with predeclared correction disturbances and risk.

**Recipient-specific prediction:** Goal-state occupancy that lies outside the empirical
viable set does not predict return; membership and recovery margin do.

**Native competitor:** Goal dwell, state reachability, UNISafe uncertainty-aware safe
set, control-barrier metrics, and simple local rollout error.

**Adoption barrier:** Computing a kernel in continuous learned latents is expensive and
may become another discretization-sensitive graph metric.

**Enabling change:** Pendulum and PointMass expose exact set_state interventions; the BH
report explicitly distinguishes reaching from dwelling.

**Transfer novelty:** Donor novelty: none. Correspondence: low-medium. Adaptation:
medium. Prediction: low-medium. Overall: N2-T; rise to N3-T only if the
uncertainty-decomposed coverage definition beats native viability/reachability baselines.

### Transfer 6 — Optimal experiment design into control-targeted excitation

**Candidate:** E1 interventional-rank coverage

**Donor field:** Optimal experimental design and system identification

**Recipient field:** Curiosity-driven world-model data collection

**Domain-neutral functional problem:** Spend a limited intervention budget to resolve
the parameter/effect directions that matter for a downstream functional.

| Structural role | Donor field | Recipient field |
|---|---|---|
| State | current system/parameter belief | latent state and ensemble model |
| Observation | experiment response | next latent/reward |
| Action/operator | excitation input/design point | feasible environment action |
| Objective/energy | information/target variance reduction | action-effect and control uncertainty reduction |
| Invariant | identifiable directions span target | feasible actions distinguish control-relevant outcomes |
| Noise/uncertainty | observation covariance | aleatoric Prediction component |
| Boundary condition | experiment budget/reachability | environment steps and current policy reach |
| Failure mode | unexcited parameter direction | planner cannot compare counterfactual actions |

**Preserved causal mechanism:** State diversity does not identify dynamics unless
interventions excite distinguishable response directions.

**Broken correspondences:**

1. Prospect's model is nonlinear/nonparametric—correctable only with local Jacobians or
   empirical contrasts.
2. Reachable design points depend on the current imperfect controller—scientifically
   productive and a source of selection bias.
3. Encoder updates change information geometry—requires a frozen audit checkpoint.
4. Fisher information can improve parameters irrelevant to control—handled by a
   control-targeted comparison, not assumed away.

**Required invention:** A lightweight, task-conditioned local action-effect statistic
that is stable enough to measure before proposing an online design policy.

**Recipient-specific prediction:** At matched state coverage, full-rank local action
data improves unseen-action prediction and control; global D-optimality can still waste
budget away from planner demand.

**Native competitor:** Random actions, action entropy, maximum epistemic, MaxInfoRL,
global D-optimal selection, and current curiosity.

**Adoption barrier:** Last-layer/Jacobian approximations may be model-specific and
counter to Prospect's narrow dependency-free protocols.

**Enabling change:** PointMass provides two action dimensions and exact placement;
Gaussian NLL supplies aleatoric weights.

**Transfer novelty:** Donor novelty: none. Correspondence: low. Adaptation: medium.
Prediction: medium only under matched state marginals. Overall: N2-T.

### Transfer 7 — Critical slowing into an integrity early-warning process

**Candidate:** Recovery-time sentinel

**Donor field:** Ecological resilience and early-warning theory

**Recipient field:** Predictive-model integrity monitoring

**Domain-neutral functional problem:** Detect loss of resilience before a coarse
performance variable tips by measuring slower recovery from standardized perturbations.

| Structural role | Donor field | Recipient field |
|---|---|---|
| State | ecosystem near an attractor | closed-loop latent/control regime |
| Observation | recovery trajectory | VoE, epistemic, return after probe |
| Action/operator | small external perturbation | standardized state/action/model perturbation |
| Objective/energy | resilience basin | supported controllable region |
| Invariant | stable system returns promptly | competent controller corrects small probes |
| Noise/uncertainty | environmental fluctuations | aleatoric variability |
| Boundary condition | quasi-stationary regime | fixed model checkpoint/local task basin |
| Failure mode | critical slowing before tipping | rising recovery time before return collapse |

**Preserved causal mechanism:** A stability margin can shrink before the endpoint
metric visibly fails, revealing itself in recovery dynamics.

**Broken correspondences:**

1. Prospect learns and replans, so it is not stationary—correctable only at frozen
   checkpoints.
2. Task trajectories need not be equilibria—limits use to repeatable local regimes.
3. Aleatoric noise can mimic rising variance/autocorrelation—requires standardized
   perturbations and uncertainty attribution.
4. Classical early-warning indicators have known false positives—scientifically
   productive and demands a matched change-point baseline.

**Required invention:** A short intervention/recovery protocol and lead-time metric,
not another passive moving average.

**Recipient-specific prediction:** Recovery time and epistemic-VoE autocorrelation rise
before control loss under gradual replay depletion or dynamics shift, outperforming
endpoint sentinels.

**Native competitor:** SurpriseCompetenceMonitor, forgetting latch, global MSE,
epistemic/error correlation, and CUSUM/change-point detection.

**Adoption barrier:** Standardized perturbations may be unsafe or unavailable outside
simulation, and the signal may duplicate ordinary local stability margins.

**Enabling change:** Authored environments permit exact small perturbations and P7
already supplies controlled nonstationarity.

**Transfer novelty:** Donor novelty: none. Correspondence: medium. Adaptation: medium.
Prediction: low-medium. Overall: N2-T measurement transfer; N3-T only if a distinct
hazard/recovery evidence unit predicts future capability.

### 7.1 Transfer-screening verdicts

- Terminology removal leaves mechanisms for Transfers 1, 2, 4, 5, 6, and 7, but the
  audit found that the first three mechanisms are already bridged much farther into
  their recipient fields than the initial generation pass assumed.
- Counter-analogy is strongest for T1's correlated moving graph and T2's shared-model
  residuals. Either can be fatal; both are therefore tested before implementation.
- Native-baseline tests are mandatory: no transfer is credited for beating only raw
  curiosity or global MSE.
- Historical-obviousness downgrades persistent excitation, viability, and adjoint
  weighting to N2-T. They remain useful baselines against which T1 must earn its place.
- Reliability/percolation is N1 at broad scope and survives only as a correlated
  finite-size scaling conjecture. Trajectory syndrome is N1 and survives only as a
  formal Prospect-head isolability benchmark. Proof/assurance is infrastructure and
  survives scientifically only as comparative counterfactual revocation. Calling
  probabilistic gate evidence a proof is explicitly rejected.
- Critical slowing is the required measurement transfer; it is not recommended ahead
  of the causal coverage fixture.

## 8. New-evidence discovery programs

These are proposed experiments, not N4 results. They were designed to discriminate
competing explanations rather than to showcase a preferred mechanism.

### Evidence program P1 — BridgeControl causal factorial

**Question:** Which property hidden inside “coverage”—visited nodes, a directed bridge,
local action excitation, or independent density—causes evidence to become control?

**Competing explanations:**

1. Node-coverage: visiting all state regions is sufficient.
2. Bridge-connectivity: observed directed transitions must connect start to goal.
3. Action-excitation: feasible action effects must be locally identifiable.
4. Density: enough independent samples per local transition is the main requirement.
5. Ordinary prediction: all structural effects disappear after controlling for global
   one-step MSE.

**Minimal prototype:** Add one deterministic NumPy fixture under bench with start and
goal basins, six intermediate regions, one directed bottleneck transition, two action
dimensions, a high-entropy reward-irrelevant nuisance coordinate, and harness-only
exact state placement. The unchanged FlatWorldModel and FlatPlanner remain the learner
and controller.

Generate a preregistered 2×2×2 factorial:

- bottleneck transition absent versus present;
- locally rank-deficient versus full-rank action sampling;
- one versus eight independent samples per state-action cell.

All primary datasets have identical state-bin counts, total transitions, nuisance
coverage, action marginals where possible, model updates, and planner compute. A
separate nuisance-only augmentation tests whether extra irrelevant coverage looks
beneficial.

**Baselines:** Random on-policy data; uniform exact-state sampling; shipped
epistemic-curiosity data; the fully balanced edge/action/density oracle dataset; and
global MSE/entropy/nearest-distance metrics.

**Controls:**

- Positive control: exact dynamics plus shipped planner must solve the task, and the
  fully balanced learned dataset must approach it.
- Structural negative control: remove exactly one bottleneck edge while visiting every
  node.
- Semantic negative control: permute action labels while preserving state histograms.
- Nuisance negative control: increase irrelevant-coordinate diversity only.
- Invariance control: repeat graph analysis at two fixed partition resolutions and in
  ground-truth coordinates; latent metrics are secondary.

**Measured variables:** Node coverage, visitation entropy, directed reachability,
minimum-cut support, local action-effect minimum singular value, unique transition
density, global and regional one-step NLL/MSE, horizon-k error, epistemic calibration,
CEM action regret, and downstream return.

**Predeclared signatures:**

- Node coverage predicts equal performance because primary arms visit the same nodes.
- Bridge mechanism produces the largest bottleneck main effect.
- Excitation produces a main effect and an interaction with bridge presence.
- Density matters mostly when bridge and excitation are already present.
- Conventional explanation predicts that global MSE absorbs all structural effects.

Use seed-blocked factorial contrasts and bootstrap intervals fixed before viewing
returns. Conductance is evaluated against the native metrics; no post-hoc best
partition is allowed.

**Kill and pivot criteria:** The fixture is invalid if exact dynamics cannot solve it or
the positive learned dataset does not. Redesign once before hypothesis testing if
factors fail to manipulate their intended diagnostics independently. Reject T1 if
ordinary global MSE fully predicts return, if the bridge contrast is within noise over
five seeds, or if it fails on a second bottleneck length. Keep the fixture as a
negative-control benchmark even after rejection.

**Confounders:** Exact resets are privileged data construction, not a proposed online
collector. Learned latent bins move, so causal variables are primary in ground-truth
coordinates. Matched action marginals do not guarantee matched state-action
conditionals; both must be reported.

**Cost:** One implementation day, fewer than 50 small NumPy trainings, about 1-3
CPU-hours after implementation.

**Execution resolution (2026-07-13, BC-001):** P1 was implemented as the non-gated
[BridgeControl fixture](../../bench/bridge_control/) under the frozen
[execution protocol](2026-07-13-bridge-control-protocol.md). The exact-model control
solved all 32 evaluation starts, while the fully balanced learned control solved only
2/32 (6.25%), below the predeclared 80% floor, after the one permitted fixture
redesign. The seven remaining factorial cells and downstream negative controls were
therefore not trained. No bridge, excitation, or density effect was estimated. The
exact-transition/learned-reward/zero-epistemic hybrid solved 27/32 starts (84.375%), a
large rescue which shows that the reward head is not the sole blocker. Because the rung
jointly changes transition, representation, and uncertainty handling, it does not
uniquely attribute the rescue. Under this portfolio's abandonment rule, T1 is retired
as the active next mechanism program; that program-level decision is not a refutation
of every transition-support hypothesis. See the
[BC-001 report](../../bench/bridge_control/results/BC-001/BC-001-report.md) and
machine-readable result for the complete stopped experiment.

**Effect on Prospect's graph:** A positive structural effect creates a measurement task
and non-gated fixture first. Only replication on a second topology justifies a
P3/P9 negative control or ADR-0008 measurement amendment. It does not directly
authorize a new collector.

### Evidence program P2 — BH paired-seed causal decomposition

**Question:** Does curiosity fail because of sparse goal density, missing corridor
support, uncertainty aversion, or generic information/capacity?

**Committed anchor:** First reproduce the pinned 4,096-step random and curious pools
under dm_control 1.0.43, MuJoCo 3.10.0, and NumPy 2.4.6. Interpretation stops if the
committed aggregates cannot be reproduced within preregistered tolerance.

**Minimal prototype:** Retain raw transitions and simulator snapshots, then construct
equal-size data arms:

- R: random pool;
- C: curiosity pool;
- M: chunk-stratified 50/50 mixture;
- G: curiosity pool with 512 transitions replaced by unique short bursts from
  discovered high-reward snapshots;
- I: matched rare/high-energy but reward-irrelevant burst augmentation;
- X: positive-control expert-corridor augmentation, 512 expert-path plus 3,584 random
  transitions.

Evaluate every arm with the shipped exploit penalty and with penalty zero. Hold model,
updates, seeds, and planner compute fixed.

**Baselines:** R, C, M, matched-budget model-free CEM-ES, and the expert-corridor
positive control X.

**Controls:** I matches rarity and count without task route; X checks that the current
model/planner can exploit adequate evidence; snapshot replay must reproduce a fixed
transition; expert actions appear only in X and never in evaluation.

**Measured variables:** Unique transition/action spread in bottom, corridor, and goal
strata; max reward; dwell; state histogram; connected segment length; one-step and
horizon-k regional errors; local epistemic/error correlation; planner truncation and
penalty fractions; support distance; action-ranking agreement; and return.

**Predeclared signatures:**

- Goal-density: G beats M and C; removing penalty has little effect.
- Corridor: M or X beats G and the gain tracks corridor error.
- Planner aversion: penalty zero rescues C/G without a mean-error change.
- Generic information: I helps about as much as G or every unique augmentation scales
  similarly.

**Kill and pivot criteria:** Stop on failed reproduction. If X improves neither regional
error nor return, move to the oracle ladder rather than tuning coverage. If all
interventions are within paired-seed noise, archive the result as “below BH resolution.”

**Confounders:** Snapshot restoration may omit hidden simulator state; penalty removal
can exploit model bias; X changes both action and state distributions; unique versus
duplicate bursts must be separated.

**Cost:** Approximately 36 three-seed train/evaluate cells, 8-16 CPU-hours plus
instrumentation, parallelizable.

**Effect on Prospect's graph:** Corridor/goal evidence supports a bounded collection
metric task; a penalty result requires ADR-0006/0007 review; rollout bias prioritizes
U-006; uncertainty signal failure prioritizes U-007/U-008. BH remains non-gated.

### Evidence program P3 — Simulator-oracle localization ladder

**Question:** Which stage—dynamics mean, reward, uncertainty arbitration, rollout
depth, or iCEM search—prevents covered experience from becoming control?

**Competing explanations:** Dynamics bias; reward-model bias; epistemic penalty or
calibration; compounding rollout error; and proposal/search failure.

**Minimal prototype:** Build a harness-only OracleWorldModel from cloned MuJoCo state
and apply nested substitutions from the same saved evaluation states:

1. shipped learned dynamics/reward/epistemic and iCEM;
2. learned model with zero uncertainty penalty;
3. learned dynamics with oracle reward;
4. oracle dynamics/reward with shipped iCEM compute;
5. oracle dynamics with enlarged-search iCEM;
6. oracle dynamics with the expert sequence injected into the candidate pool;
7. oracle transitions for the first k in {1, 2, 4, 8, 20} steps, then the learned model;
8. learned mean with privileged actual-error gating, for diagnosis only.

**Baselines and controls:** Shipped MBRL, matched CEM-ES, random, and expert/oracle-clone
ceilings. Oracle dynamics plus expert candidate must solve the task. Permuted oracle
reward and an oracle applied only to a nuisance coordinate are negatives. Cloned state
and identical action sequence must reproduce transitions within strict tolerance.
Planner candidate compute is identical except for the explicitly labeled enlarged arm.

**Measured variables:** Return; chosen-action regret; candidate-ranking agreement;
state/reward error by depth; penalty contribution; fraction of failure removed by each
substitution; and minimum oracle-prefix length for recovery.

**Predeclared signatures:** Dynamics rescue begins at oracle dynamics; uncertainty
rescue begins at penalty removal/privileged gating with stable mean rankings; reward
rescue begins at oracle reward; search rescue requires enlarged search or expert
injection even under exact dynamics; compounding error shows a reproducible knee over k.

**Kill and pivot criteria:** Abort if the positive oracle pipeline cannot solve
swing-up. If cloning is slow/nondeterministic, use an offline action-ranking ladder at
short horizons. If no substitution has a stable paired effect, repeat only once at a
predeclared doubled budget and then classify the probe as below causal resolution.

**Confounders:** Oracle state includes hidden variables absent from observations;
oracle epistemic is privileged and cannot become a runtime mechanism; simulator and
planner compute must be reported separately.

**Cost:** Roughly 6-12 CPU-hours if cloning is batched; up to two workstation-days if
unoptimized.

**Effect on Prospect's graph:** Dynamics/depth fires U-006 evidence; uncertainty fires
U-007/U-008 before U-105; reward creates a new bounded integrity task; search creates a
planner proposal task only after U-101/U-104 trigger analysis.

### Evidence program P4 — Identical-data order and hysteresis

**Question:** Is coverage-to-control failure a property of the final evidence multiset,
or of the path by which the model learned and collected it?

**Competing explanations:** Final dataset composition; representation/EMA hysteresis;
recency/forgetting; and closed-loop policy-data feedback.

**Minimal prototype:** Generate one fixed random and one fixed curiosity pool. Hash the
raw transition multiset and train identical models with identical update/minibatch
budgets under random-then-curious blocks, reverse blocks, alternating chunks, one
global shuffle, and repeated full-union shuffles. Then repeat a smaller set online,
where intermediate models control future collection.

**Baselines and controls:** Random-only, curiosity-only, and 50/50 shuffled union.
Closed-form linear regression on the same union is an order-invariant negative control;
a deliberately last-block-only learner is a sensitivity positive control. Precomputed
minibatch index schedules isolate order from SGD randomness.

**Measured variables:** Return after each block and at completion; regional one-step
and rollout errors; effective rank; encoder/EMA drift; epistemic/error correlation;
between-checkpoint prediction disagreement; and forgetting/relearning curves.

**Predeclared signatures:** Composition makes all mixed offline schedules converge;
hysteresis leaves different final predictions despite identical hashes; recency favors
the last regime and disappears under full shuffle; closed-loop feedback appears only
online because transition multisets diverge.

**Kill and pivot criteria:** Reject any run with unequal data/update exposure. Attribute
effects to SGD noise and stop if deterministic minibatches remove them. Treat effects
only at an extreme learning rate as an instability region, not a new coverage law.

**Confounders:** Target-encoder state makes order genuinely causal; normalization must
be frozen/logged; current 4,096-transition pools do not exercise reservoir eviction.

**Cost:** Five schedules × five seeds plus a three-schedule online follow-up, about
3-6 CPU-hours and one implementation day.

**Effect on Prospect's graph:** Stable representation hysteresis could fire U-102;
non-relearning may justify a plasticity gate and later U-107; order invariance rules out
training history and prioritizes geometry/oracle experiments.

## 9. Pre-execution Pareto frontier and post-execution disposition

These scores were frozen before BC-001 outcomes. They are independent 0-5 judgments;
5 is better. “Test economy” means lower first-test cost. They do not establish novelty
or combine into a single ranking, and the execution result supersedes candidate status.

| Candidate | Apparent novelty | Falsifiability | Explanatory value | Importance | Feasibility | Test economy | Result interpretability | Baseline strength | Useful negative result | Publication potential |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 correlated control reliability (retired after BC-001) | 2.0 | 5.0 | 5.0 | 5.0 | 4.5 | 5.0 | 5.0 | 5.0 | 5.0 | 3.5 |
| T2 Prospect-head diagnosability | 1.0 | 4.5 | 3.5 | 3.5 | 3.5 | 4.0 | 4.5 | 4.5 | 4.5 | 2.5 |
| **T3 counterfactual evidence revocation** | 1.0 | 5.0 | 3.0 | 4.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 2.0 |
| **T4 job-compatibility polytope** | 3.0 | 4.0 | 5.0 | 5.0 | 3.5 | 4.0 | 3.5 | 4.0 | 4.5 | 4.0 |
| R1 plan-demanded holes | 2.0 | 4.0 | 4.0 | 4.0 | 4.0 | 3.0 | 4.0 | 4.0 | 4.0 | 3.0 |
| **R2 decision-flip epistemic** | 2.0 | 4.5 | 4.0 | 4.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 3.0 |
| **E1 interventional-rank coverage** | 2.0 | 5.0 | 4.0 | 4.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 3.0 |
| E2 conditional validity field | 1.5 | 4.0 | 3.0 | 3.0 | 4.0 | 5.0 | 4.0 | 4.0 | 4.0 | 2.5 |
| E3 dynamic competence stock | 2.5 | 4.0 | 3.5 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 3.0 |
| E4 projective family audit | 1.0 | 4.0 | 2.5 | 3.0 | 5.0 | 5.0 | 4.0 | 5.0 | 4.0 | 2.0 |

The bold rows formed the pre-execution retained Pareto set for different objectives:

- T1 had the highest causal explanatory value per first-test cost, but BC-001 then hit
  its predeclared abandonment criterion; it is no longer active.
- T2 is retained as a diagnostic benchmark but is dominated on novelty and feasibility.
- T3 is the easiest way to improve the reliability of all later evidence, while being
  infrastructure rather than a broad research primitive.
- T4 directly stress-tests Prospect's defining one-signal-many-jobs invariant.
- R2 and E1 were cheap native explanations that T1 had to beat; no contrast was
  estimated, so they remain unresolved measurements rather than winners.

Post-execution, T3, T4, R2, and E1 remain active candidates at their audited novelty
levels. P3, the simulator-oracle ladder, is now the active evidence program rather than
a mechanism: it is the declared fallback after the designated balanced arm did not
rescue the BC-001 controller.

## 10. Recommended first experiment — executed and stopped

### BridgeControl: a one-day causal coverage fixture

**Recommendation:** Implement only Evidence Program P1 under bench as a non-gated
research fixture. Do not first implement predictive conductance, a new collector, or a
production API. The fixture is smaller and can kill T1, E1, and ordinary density
explanations in one experiment.

**Resolved:** BC-001 was run on 2026-07-13 and stopped at its learned positive control.
The recommendation below is retained as the preregistered rationale, not as pending
work. The [execution report](../../bench/bridge_control/results/BC-001/BC-001-report.md)
records the negative result and simulator-oracle pivot.

**Why this experiment first:** The six-row BH computation is too confounded to choose
between data topology, action excitation, local density, uncertainty arbitration, and
planner search. BridgeControl deliberately holds point coverage constant and
independently changes the first three. Existing exact-state harness conventions make
that manipulation cheap; any method built before it would encode an untested answer.

**Null hypothesis:** Conditional on matched state/action marginals, transition budget,
training updates, and global one-step error, bottleneck-edge presence and local
action-effect rank add no reproducible information about MPC return.

**Mechanism signature:** Return exhibits a positive bridge main effect and a
bridge-by-action-rank interaction; the minimum reliable cut or its simpler
manipulation-check proxy predicts return better than node coverage, entropy, global
MSE, mean epistemic, and nearest-neighbor distance. Nuisance-only coverage has no
benefit.

**Strongest conventional signature:** Global one-step MSE absorbs the structural
contrasts, or local density alone explains them. In that case reject predictive
conductance and use the fixture to calibrate ordinary model-error requirements.

**Exact first-pass design:**

1. Author a deterministic two-action fixture with a known bottleneck and nuisance
   coordinate, plus exact state placement.
2. Validate exact dynamics with the unchanged FlatPlanner before generating learned
   datasets.
3. Generate all eight 2×2×2 fixed datasets before training, hash them, and publish
   manipulation summaries without returns.
4. Train the unchanged FlatWorldModel for identical updates under seeds 0-4.
5. Evaluate identical iCEM compute and a fixed held-out transition/initial-state set.
6. Analyze only preregistered main effects/interactions and metric comparisons.
7. Repeat the bridge contrast on one second bottleneck length only if the first
   manipulation checks and positive controls pass.

**Named native baselines:** Node visitation count, visitation entropy, maximum reached
reward, global one-step MSE/NLL, regional bridge MSE, mean and accumulated epistemic,
nearest-neighbor support distance, local action entropy, local action-effect minimum
singular value, and exact directed reachability.

**Named literature baselines:** ROVER-style occupancy coverage, persistent-excitation
rank, viability/reachability, and value/decision-aware error. These can be implemented
as measurements on fixed datasets; no competing full agents are required initially.

**Metrics and plots:**

- factorial return interaction plot with every seed visible;
- node/edge occupancy and action-condition manipulation table;
- global and region-stratified one-step/horizon-k error;
- chosen-action regret and candidate-ranking agreement;
- return versus each preregistered proxy with leave-one-seed-out fits;
- graph visualization marking the bottleneck and minimum cut;
- positive/negative-control outcome table.

**Leakage, bug, and noise checks:**

- Ground-truth coordinates define primary cells; the learned latent is reported only as
  a secondary robustness view.
- Dataset generation may use exact resets, but training and evaluation receive only
  ordinary transitions/observations.
- Action labels and next states are hashed before training; the permuted-action control
  must preserve state histograms and fail control.
- Exact-model planning and the fully balanced learned dataset must succeed.
- A coordinate/nuisance permutation that preserves dynamics must not change the
  conclusion.
- Training, planner candidates, updates, and evaluation starts are identical per seed.

**Decisive observation:** A bridge effect that is larger than the nuisance/density
control, has the same sign in all five seed blocks, retains a bootstrap interval above
zero, and replicates at the second bottleneck length while global MSE remains matched.
Thresholds for “matched” diagnostics and positive-control success must be fixed during
fixture validation, before learned-return inspection.

Even that observation would establish only a causal bridge effect. It would license,
not complete, the narrow research program: correlated reliability must then predict
threshold location and finite-size scaling across topology, budget, and partition
resolution better than coverability, sequence coverage, bottleneck centrality, and
conformal reachability.

**Cost:** One implementation day and 1-3 CPU-hours. No external GPU, production code,
ADR, task status, gate result, or shipped artifact changes.

**Abandonment criterion:** Abandon T1 after one fixture redesign if the factors cannot
be manipulated independently, if exact/balanced positive controls fail, if global MSE
fully mediates return, or if the bridge effect fails the second topology. Do not rescue
it with post-hoc bins, a learned gate, or coefficient tuning.

**Useful artifact under failure:** A compact causal negative-control benchmark for
future coverage metrics, action-identification claims, and mutation of P3/P9 evidence.
It also tells the next branch: oracle ladder for pipeline/search failure, or
interventional-rank work if excitation rather than bridge structure survives.

## 11. Audit limitations

### 11.1 Final novelty disposition

The adversarial pass materially changed the portfolio:

- Broad reliability-percolation/control-cut coverage is rejected as N1. Only a narrow
  N2-T correlated threshold and finite-size scaling conjecture remains. Its hypothesis
  was generated by new repository evidence (N4-T provenance), but that does not make
  the imported formulation conceptually new.
- Broad continuous-trajectory syndrome theory is rejected as N1. A Prospect-head
  diagnosability/isolability matrix is a possible narrow N2 adaptation.
- Broad proof/artifact-bound evidence is rejected as N0-N1 infrastructure. A
  counterfactual revocation/minimal-witness operator is possible narrow N2 only after a
  comparison with CAF-style assertions.
- T4, the job-compatibility polytope, remains provisional N3 but did not receive the
  same exhaustive donor/recipient audit; direct decision-aligned-UQ work is a serious
  threat.

Therefore this portfolio does not claim a surviving major novelty discovery. It
delivers a reusable prompt, a new hypothesis-generating computation, a falsifiable
portfolio, three explicit novelty destructions, and a stopped causal assay whose
negative result selects the simulator-oracle evidence branch.

### 11.2 Representative query log

Searches used exact recipient terms, donor terms, bridge descriptions, synonyms, and
older terminology. Representative queries were:

- “world model exploration graph connectivity coverage control latent transitions
  bottleneck”
- “transition coverage reinforcement learning downstream control dataset graph”
- “coverage bottleneck exploration reinforcement learning transition graph world
  model”
- “world model as graph reachability edge probability path cut search”
- “parity space residual syndrome fault isolation structured residuals diagnosis
  neural dynamics”
- “world models failure diagnosis error attribution uncertainty ensemble disagreement
  causes”
- “world model multi-timescale consistency prediction reconciliation hierarchical
  forecasts control”
- “value-aware model learning reinforcement learning model error value gradient
  control”
- “active system identification persistent excitation reinforcement learning
  exploration dynamics model”
- “decision-aligned uncertainty evaluation downstream utility calibration”
- “proof-carrying artifact machine-readable AI assurance claim evidence DAG”
- “machine learning benchmark artifact provenance mutation testing sentinels same
  model”
- “SLSA provenance artifact attestation subject predicate official specification”
- “mutation testing deep reinforcement learning real faults gate adequacy”

The repository's July-2026 SOTA review expanded terms but was not treated as proof of
novelty or correctness.

### 11.3 Primary-source classes and strongest audit sources

| Source class | Representative primary sources | Role in audit |
|---|---|---|
| Coverage and graph control | [L3P](https://proceedings.mlr.press/v139/zhang21x.html), [adaptive-grid exploration](https://openreview.net/forum?id=59MYoLghyk), [coverability](https://proceedings.mlr.press/v235/amortila24a.html), [sequence coverage](https://proceedings.mlr.press/v286/zhou25a.html), [ROVER](https://arxiv.org/abs/2606.21271) | destroyed broad graph/coverage novelty |
| Reachability and path/cut | [UNISafe](https://arxiv.org/abs/2505.00779), [path-and-cut feasibility](https://arxiv.org/abs/2305.10395), [DBAP](https://dbap-rl.github.io/) | showed calibrated reachability and explicit cut reasoning already exist |
| Model exploitation/decision utility | [Imperfect World Models are Exploitable](https://arxiv.org/abs/2605.15960), [policy-aware WMs](https://policy-aware.github.io/paper-anon/), [decision-aligned UQ](https://arxiv.org/abs/2606.26990), [VaGraM](https://arxiv.org/abs/2204.01464) | supplied policy-ranking and utility-aware native alternatives |
| Fault diagnosis and consistency | [neural residual isolation](https://arxiv.org/abs/1910.05626), [data-driven residual diagnosis](https://arxiv.org/abs/2305.04670), [world-action consistency](https://arxiv.org/abs/2605.07514), [WFM-Eval](https://openreview.net/forum?id=gvtwynIibB) | destroyed broad syndrome novelty |
| Evidence and assurance | [in-toto](https://www.usenix.org/conference/usenixsecurity19/presentation/torres-arias), [SLSA](https://slsa.dev/spec/v1.2/provenance), [AMLAS](https://arxiv.org/abs/2102.01564), [CAF](https://ojs.aaai.org/index.php/AAAI/article/view/41151), [muPRL](https://arxiv.org/abs/2408.15150) | destroyed broad artifact/proof novelty |
| Cross-timescale coherence | [probabilistic forecast reconciliation](https://arxiv.org/abs/2103.11128) | downgraded projective prediction family |
| Experimental design | [persistent-excitation active ID](https://openreview.net/forum?id=zlSHHzj3AU), [localized dynamics learning](https://openreview.net/forum?id=j6mc0t9Ltsj) | downgraded action excitation and plan-demand ideas to transfers/recombinations |

ArXiv/OpenReview/workshop sources establish prior art but not equal validation status.
The audit distinguishes existence from empirical strength.

### 11.4 Inaccessible or incomplete evidence

- Two OpenReview forum pages for MTS3 and MCPlanner redirected to browser challenges,
  so only indexed metadata and repository summaries were available; E4's downgrade is
  conservative.
- Some classical parity-space, network-reliability, and surrogate-dynamics works were
  accessible only through abstracts/proceedings or publisher metadata, not complete
  paywalled text.
- This was not a patent search, systematic review, citation-network closure, exhaustive
  thesis search, or exhaustive non-English search. Novelty confidence must not be used
  for a patentability or publication-priority claim.
- T4 and minor candidates received targeted rather than patent-grade audits. A second
  adversarial pass is required before writing an ADR or paper around them.

### 11.5 Empirical limitations

- BC-001 is the only new training experiment in this portfolio update. It is a
  non-gated fixture; it does not amend P0-P14 or BH shipped claims. Repository gates
  were rerun only as regression checks, not as new research evidence.
- The earlier six-row JSON reanalysis is also new empirical evidence. It is small,
  confounded by collection arm, and cannot distinguish mediation, causality, or
  within-arm monotonicity.
- BridgeControl's exact control passed but its fully balanced learned control failed
  after the one allowed redesign. The causal factorial was stopped, so all bridge,
  excitation, density, interaction, and scaling effect sizes remain unestimated. The
  84.375% exact-transition/learned-reward/zero-epistemic hybrid is a localization
  diagnostic that jointly changes transition, representation, and uncertainty
  handling; it is not a causal estimate or a unique attribution.
- Edge independence, graph partition stability, action-effect estimation, and viable
  goal tests are unresolved. A positive one-topology bridge contrast would not
  establish percolation or a general reliability law.
- No candidate may change source, gates, tasks, ADRs, backlog status, or shipped results
  on the strength of this document alone.

### 11.6 Terminology and category risks

- “Coverage” spans occupancy, concentrability, state-action support, sequence support,
  reachability, and test coverage. Every future task must name the exact object.
- “Conductance” and “percolation” have established mathematical meanings. Until a
  dependence-aware scaling result exists, “correlated control reliability” is the safer
  label.
- “Syndrome” implies a diagnosability code. Without unique failure signatures, use
  “structured residual benchmark.”
- “Proof” is inappropriate for empirical generalization. Artifact integrity and
  statistical evidence can be checked; downstream capability remains conditional.
- Structural validation of this Markdown file checks completeness and calibrated
  wording only. It does not establish novelty, correctness, or publication value.

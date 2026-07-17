# Research Portfolio: Causally witnessed experience-to-retention learning

**Repository/domain:** Prospect; adaptive predictive agents and agent-learning
evaluation<br>
**Literature cutoff:** 2026-07-17<br>
**Sources searched:** active repository source, tests, ADRs, task and research
records; arXiv; PMLR; NeurIPS; AAAI; OpenReview; JMLR; NIST/JCGM; FDA guidance;
database recovery literature; official project repositories and documentation<br>
**Search scope:** targeted title, synonym, functional-description, donor-field, and
bridge-field search; not a patent, thesis, or complete citation-network review<br>
**Key unresolved assumptions:** the first real learner and task family are not yet
selected; intended publication venue, compute budget, and acceptable online
evaluation risk are unspecified; the current finite benchmark is a semantic oracle,
not a learning result

The research question is:

> Can one running agent make a persistent behavioral change whose causal ancestry
> in exact experience is independently checkable, whose promotion is justified by
> disjoint real outcomes, and whose gain survives genuine interference and process
> restoration?

The current repository does not answer that question. This portfolio separates the
minimum experiment needed to answer it from speculative research contributions.
All novelty classifications are provisional under the search disclosed above.

## 1. Frontier map

| Region | Primitive and mechanism | Evidence already established elsewhere | Prospect's unresolved edge |
|---|---|---|---|
| Experience-based language agents | trajectories, textual lessons, retrieval, reflection, skill libraries | [ExpeL](https://ojs.aaai.org/index.php/AAAI/article/view/29936), [Reflexion](https://papers.nips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html), and [Voyager](https://arxiv.org/abs/2305.16291) show several forms of experience reuse | exact causal attribution, statistical controls, shared-state interference, and fresh-process retention in one chain |
| Learned adaptation and harnesses | learned reflector/adaptation policy or editable harness | [Training Language Agents to Learn from Experience](https://arxiv.org/abs/2605.20477), [ExpWeaver](https://openreview.net/forum?id=4ARH5kZrgz), and the very recent [MemoHarness](https://arxiv.org/abs/2607.14159) directly threaten broad “learns from experience” claims | evidence-bearing promotion and retention, rather than another memory or harness mechanism |
| Information-seeking decisions | posterior distributions, EIG, EVSI, expected free energy, active experiment choice | mature Bayesian experimental design, active learning, active inference, and curiosity methods | calibration of predicted information value against realized decision value under misspecification, trust, and cost |
| Continual RL/world models | shared parameters, replay, regularization, task streams, forgetting and transfer | [Continual World](https://arxiv.org/abs/2105.10919) and [Continual-Dreamer](https://proceedings.mlr.press/v232/kessler23a.html) provide task streams and strong continual-learning baselines | same-chain experience custody and production restart as part of the scientific claim |
| Safe improvement | incumbent/candidate policies, confidence bounds, constrained promotion | [Safe Policy Iteration](https://www.jmlr.org/papers/v22/19-707.html) and related HCOPE/SPI work constrain policy replacement | extending promotion to heterogeneous model, memory, harness, calibration, and retrieval updates with retention co-endpoints |
| Event-sourced agents and recovery | append-only events, projections, forks, deterministic replay, WAL | event sourcing and database recovery are mature; [The Log is the Agent](https://arxiv.org/abs/2605.21997) directly applies log-centric execution to agents | logs reconstruct what happened but do not by themselves prove that an update caused a valid improvement |
| Proof-carrying governance | action envelope, proof obligation, checker, authorization receipt | [Proof-Carrying Agent Actions](https://arxiv.org/abs/2606.04104) binds action authorization and outcome closure across runtimes | statistical certificates for causal learning, disjoint improvement, interference, and retention |
| Measurement science | measurand, calibration, traceability chain, uncertainty budget, validity domain | [Metrology for AI](https://arxiv.org/abs/1911.01875), the [JCGM GUM](https://www.iso.org/sites/JCGM/GUM-introduction.htm), and NIST traceability practice | making a scoped, expiring capability/knowledge measurement govern live agent-state reuse |
| Current Prospect | typed evidence graph, exact finite epistemics, authoritative step runtime, canonical stores, checkpoint components | active structural and semantic tests pass | no real same-chain learner, disjoint executed behavior, shared interference, or complete fresh-process restore |

### Dense and sparse regions

Dense regions include adding memory, reflection, retrieval, replay, curricula,
skills, uncertainty bonuses, larger world models, and checkpoint serialization.
Those mechanisms may be useful substrates but are poor default novelty targets.

The apparently sparse intersection is narrower: causal experience ancestry,
versioned persistent change, independently measured behavioral gain, adversarial
counterfactual controls, genuine shared-state interference, and process
restoration all belonging to one promoted update. No exact match was found in the
targeted search, but the neighboring 2026 work makes this a fragile, provisional
observation rather than a novelty conclusion.

### Residuals and negative evidence in Prospect

- The E2 collector and E3 learner are different agents.
- E3 performs exact belief assimilation without changing a predictive model.
- E4 calculates expected utility instead of executing actions and outcomes.
- E5 uses isolated task slots and an incomplete benchmark checkpoint.
- A journal can expose partial work without being able to recover it.
- A checkpoint can round-trip selected state while omitting the state responsible
  for the claimed improvement.
- Low uncertainty, correct serialization, and training loss can all coexist with
  no demonstrated behavioral learning.

## 2. Functional problem signature

Let

\[
F:(H_t,S_t,G_t,B_t)\rightarrow(a_t,S_{t+1},W_t)
\]

where:

- \(H_t\) is a causally ordered history of observations, intended and executed
  actions, outcomes, provenance, identities, versions, and resource use;
- \(S_t\) is persistent belief, predictive model, policy, memory, optimizer,
  calibration, configuration, and recovery state;
- \(G_t\) contains current goals and hard constraints;
- \(B_t\) contains compute, time, risk, action, and information budgets;
- \(a_t\) may be an external action, observation request, experiment, retrieval,
  training update, abstention, or state promotion; and
- \(W_t\) is evidence witnessing why an action or persistent state change was
  admissible.

The hidden variables are environment state, task structure, sensor reliability,
unobserved counterfactual outcomes, and the causal contribution of individual
experiences to a later behavior. Local operations occur per step or update;
global claims span a complete training/evaluation/interference/restart lineage.
Discrete identities and versions coexist with continuous beliefs, parameters,
scores, and utilities. Partial observability, misspecification, correlated
evaluation noise, nonstationarity, and finite budgets limit identifiability.

A mature learning claim requires one continuous chain:

\[
S_1=U(S_0,E_{\mathrm{train}})
\]

\[
J(S_1,D_{\mathrm{heldout}})-J(S_0,D_{\mathrm{heldout}})>\delta
\]

and, after genuine shared-state interference \(I_B\) and process restoration
\(R\),

\[
J(R(I_B(S_1)),D_A)-J(S_0,D_A)>\delta_{\mathrm{retain}}.
\]

The exact experience identities, update version, held-out outcomes, controls,
interference, checkpoint, and restored behavior must belong to the same chain.
Information value remains one decision term rather than the reward definition:

\[
Q(a)=\mathbb E[U_{\mathrm{external}}]
     +\beta\,\operatorname{EVSI}(a)
     -\lambda\,\operatorname{Risk}(a)
     -\operatorname{Cost}(a).
\]

### Assumption-surgery lane

| Default assumption | Surgery | Candidate consequence |
|---|---|---|
| An optimizer may mutate live state and report afterward | reverse causality: an update first creates a quarantined candidate whose evidence must authorize promotion | causal update certificate |
| Knowledge is an internal representation with low uncertainty | infer knowledge as an external, scoped measurement with a traceability and expiry contract | metrologically traceable knowledge |
| Retention is one before/after forgetting number | model retention as a response surface over overlap, update magnitude, replay, delay, and restart | interference spectroscopy |
| More EIG is intrinsically useful | score EIG/EVSI as forecasts whose sign and calibration can fail | VOI scorecard and sign-reversal atlas |
| An experience is immutable once learned from | permit evidence retraction and require consequences to be recomputed or invalidated | revocable experience |
| One stored substrate is the capability | represent competence redundantly across heterogeneous substrates and test parity | error-correcting competence code |

### Primitive/grammar-invention lane

Four proposed primitives survive initial screening:

1. `CandidateUpdate + CausalUpdateCertificate`, replacing privileged mutation by
   evidence-governed promotion.
2. `CapabilityMeasurand`, replacing context-free “knowledge” by a scoped,
   calibrated, expiring measurement.
3. `InterferenceResponseOperator`, replacing a point retention score by a
   perturbation-to-behavior map.
4. `CompetenceSyndrome`, replacing isolated memory/model/skill state by
   cross-substrate parity constraints.

Only experiments can promote these from N3/N3-T hypotheses. None is an N4 result
because no new empirical phenomenon has yet been observed.

## 3. Fixation anti-library

The following are useful components or baselines, not sufficient research
contributions:

- add a vector database, episodic memory, RAG, or distilled lesson store;
- add reflection, self-critique, a learned prompt, or a learned harness;
- add an automatic curriculum or executable skill library;
- maximize surprise, entropy reduction, EIG, or expected free energy;
- add replay, EWC, LoRA, modular heads, or generative replay;
- build a larger world model, ensemble, or uncertainty head;
- event-source everything and call the log the learning mechanism;
- save model weights and call restoration retention;
- treat training loss, confidence, or one benchmark score as knowledge;
- add generic provenance, a knowledge graph, or cryptographic hashing;
- tune thresholds after observing the formal evaluation;
- use task names, IDs, or split metadata that leak the target rule; or
- create a custom collector, tensor batch, replay buffer, optimizer, or
  checkpoint format when a mature dependency satisfies the contract.

## 4. Productive recombinations

These should be built or used as baselines. They are not currently strong novelty
claims.

### Candidate R1 — Same-chain real learner

**Central claim:** A small shared-parameter learner can consume the exact
experiences selected by Prospect and cause disjoint predictive and executed
behavioral improvement.<br>
**Novelty class:** N1.<br>
**Known foundation:** ordinary supervised/model-based learning, replay, held-out
evaluation, and provenance.<br>
**Irreducible delta:** one identity chain joins collection, consumed samples,
version-changing update, behavior, interference, and restore.<br>
**Why this is not merely A + B:** It is a deliberate A+B foundation; it should not
be marketed as transformational.<br>
**Changed grammar or transfer mechanism:** none; it makes the current claim
measurable.<br>
**New prediction:** correct-link training beats no-update and
marginal-preserving-link permutation on disjoint outcomes.<br>
**Cheapest killing test:** a two-feature online binary or small continuous
prediction task with opaque task IDs and one shared MLP.<br>
**Prior-art threats:** standard ML pipelines, continual-learning harnesses, and ML
provenance.<br>
**Novelty confidence:** 0–10% as a contribution; cutoff 2026-07-17.<br>
**Scientific value:** indispensable foundation and negative-control generator.<br>
**Publishable if successful:** primarily as infrastructure or benchmark evidence.<br>
**Publishable if partially successful:** a failure taxonomy for causal custody.<br>
**Publishable if it fails informatively:** evidence that Prospect's contracts
overconstrain or fail to connect an ordinary learner.

### Candidate R2 — Established experience-agent adapter

**Central claim:** Prospect's evaluator can distinguish true held-out improvement
and retention from apparent gains in an existing experience-based agent.<br>
**Novelty class:** N1.<br>
**Known foundation:** ExpeL, Reflexion, MemoHarness, ExpWeaver, and their native
evaluations.<br>
**Irreducible delta:** apply unchanged causal and persistence controls to a foreign
experience mechanism.<br>
**Why this is not merely A + B:** It is A+B unless the evaluator exposes a
previously unmeasured failure law.<br>
**Changed grammar or transfer mechanism:** none; foreign methods remain the
learning mechanism.<br>
**New prediction:** some reported gains will not survive linkage permutation,
shared interference, or fresh-process component ablation.<br>
**Cheapest killing test:** adapt one small ExpeL-like textual lesson store to two
held-out task families and restart it.<br>
**Prior-art threats:** MemoHarness component attribution and standard ablation
practice.<br>
**Novelty confidence:** 5–15%; cutoff 2026-07-17.<br>
**Scientific value:** external validity and a strong native baseline.<br>
**Publishable if successful:** benchmark/audit contribution.<br>
**Publishable if partially successful:** a reproducibility report.<br>
**Publishable if it fails informatively:** evidence that ordinary evaluation
already captures all Prospect-specific distinctions.

### Candidate R3 — Calibrated VOI regulator

**Central claim:** Calibrating predicted EIG/EVSI against realized posterior
change, utility, and regret improves information-action selection under
misspecification.<br>
**Novelty class:** N2.<br>
**Known foundation:** Bayesian experimental design, active inference, calibration,
and proper scoring.<br>
**Irreducible delta:** a unified estimator scorecard tied to realized
decision-relevant value.<br>
**Why this is not merely A + B:** It remains N2 unless it yields a new, stable
failure boundary.<br>
**Changed grammar or transfer mechanism:** information value becomes a fallible
forecast rather than an oracle reward.<br>
**New prediction:** a calibrated lower-confidence VOI rule dominates raw EIG when
sensor trust and model likelihood are misspecified.<br>
**Cheapest killing test:** finite hidden-variable worlds crossed by sensor
reliability, relevance, cost, and shift.<br>
**Prior-art threats:** value-of-information calibration, Bayesian regret, and
expected-free-energy criticism.<br>
**Novelty confidence:** 15–30%; cutoff 2026-07-17.<br>
**Scientific value:** clarifies when uncertainty-directed action helps.<br>
**Publishable if successful:** estimator calibration and failure-boundary study.<br>
**Publishable if partially successful:** negative map of regimes where calibration
is unnecessary.<br>
**Publishable if it fails informatively:** evidence that standard EVSI already
absorbs the proposed correction.

### Candidate R4 — Transactional lifecycle recovery

**Central claim:** WAL-style prepare/commit/replay semantics can make every
lifecycle boundary idempotently recoverable without duplicating real experience.<br>
**Novelty class:** N1/N2-T.<br>
**Known foundation:** database write-ahead logging, event sourcing, sagas, and
checkpoint recovery.<br>
**Irreducible delta:** reconcile nondeterministic learners and external
environment effects with the canonical experience ledger.<br>
**Why this is not merely A + B:** It is systems transfer unless external
side-effect reconciliation forces a new protocol.<br>
**Changed grammar or transfer mechanism:** preserve prefix consistency: no
successful state may depend on an uncommitted or duplicated experience.<br>
**New prediction:** crash injection at every boundary yields one of two outcomes:
exact idempotent continuation or explicit compensating failure, never silent
double learning.<br>
**Cheapest killing test:** deterministic failpoints around
decide/execute/store/assimilate/learn/commit.<br>
**Prior-art threats:** ARIES-style recovery, event sourcing, workflow engines, and
The Log is the Agent.<br>
**Novelty confidence:** 5–20%; cutoff 2026-07-17.<br>
**Scientific value:** necessary production substrate and excellent fault
diagnostic.<br>
**Publishable if successful:** agent-runtime recovery protocol if genuine
agent-specific mismatches remain.<br>
**Publishable if partially successful:** an engineering pattern and conformance
suite.<br>
**Publishable if it fails informatively:** proof that standard transactional
workflow semantics suffice.

### Candidate R5 — Continual-learning benchmark adapter

**Central claim:** Prospect's same-chain evidence can be layered on an established
continual-learning task stream without altering its native plasticity, forgetting,
and transfer metrics.<br>
**Novelty class:** N1.<br>
**Known foundation:** Continual World, Continual-Dreamer, Avalanche-style metrics,
and shared-parameter interference.<br>
**Irreducible delta:** checkpoint/process identity and exact consumed-experience
ancestry supplement the native metric suite.<br>
**Why this is not merely A + B:** It is A+B unless custody explains failures the
native metrics cannot localize.<br>
**Changed grammar or transfer mechanism:** none.<br>
**New prediction:** two agents with equal forgetting scores may differ in whether
their retained behavior can be causally attributed and reproduced after restart.<br>
**Cheapest killing test:** two short MiniGrid/MiniHack tasks with a shared network,
replay and no-replay arms.<br>
**Prior-art threats:** continual-learning experiment managers and provenance
systems.<br>
**Novelty confidence:** 5–15%; cutoff 2026-07-17.<br>
**Scientific value:** avoids a bespoke toy becoming the only evidence.<br>
**Publishable if successful:** benchmark extension.<br>
**Publishable if partially successful:** integration report.<br>
**Publishable if it fails informatively:** evidence that native continual metrics
already establish the full claim.

## 5. Exploratory candidates

### Candidate E1 — Epistemic-estimator scorecard

**Central claim:** EIG, EVSI, sensor trust, and predicted uncertainty reduction
have measurably different calibration and sign-error regimes.<br>
**Novelty class:** N2.<br>
**Known foundation:** proper scoring, forecast calibration, Bayesian experimental
design, and decision regret.<br>
**Irreducible delta:** treat epistemic quantities themselves as forecasts to be
scored.<br>
**Why this is not merely A + B:** it is exploratory until a new invariant or
failure boundary appears.<br>
**Changed grammar or transfer mechanism:** the agent does not “have” correct VOI;
it issues a falsifiable VOI forecast.<br>
**New prediction:** high predicted EIG can remain calibrated for posterior change
while being anti-calibrated for decision value.<br>
**Cheapest killing test:** a finite crossed-factor atlas with exact ground truth.<br>
**Prior-art threats:** decision calibration and Bayesian regret decomposition.<br>
**Novelty confidence:** 20–35%; cutoff 2026-07-17.<br>
**Scientific value:** prevents uncertainty reduction from silently becoming
reward.<br>
**Publishable if successful:** taxonomy plus benchmark of estimator failure.<br>
**Publishable if partially successful:** calibration protocol.<br>
**Publishable if it fails informatively:** evidence that existing forecast
calibration fully predicts realized decision value.

### Candidate E2 — Causal experience-attribution graph

**Central claim:** approximate deletion/reweighting effects can identify which
experience groups caused a held-out behavioral change.<br>
**Novelty class:** N2.<br>
**Known foundation:** influence functions, data valuation, causal inference,
retraining ablations, and provenance graphs.<br>
**Irreducible delta:** bind attribution estimates to an agent's longitudinal
decision/update/retention graph.<br>
**Why this is not merely A + B:** it remains A+B unless temporal interactions yield
a distinct causal operator.<br>
**Changed grammar or transfer mechanism:** experience ancestry becomes weighted
causal contribution rather than binary lineage.<br>
**New prediction:** high-replay-frequency experiences need not have high causal
effect on retained behavior.<br>
**Cheapest killing test:** compare approximations against exact leave-group-out
retraining on the small shared learner.<br>
**Prior-art threats:** training-data attribution, Shapley data valuation, and
machine unlearning.<br>
**Novelty confidence:** 10–25%; cutoff 2026-07-17.<br>
**Scientific value:** might localize harmful experience rather than merely detect
regression.<br>
**Publishable if successful:** validated longitudinal attribution method.<br>
**Publishable if partially successful:** a map of approximation failure.<br>
**Publishable if it fails informatively:** evidence that exact lineage is useful
but causal weights are not identifiable.

### Candidate E3 — Interference spectroscopy

**Central claim:** retention and plasticity exhibit a reproducible response
surface over task overlap, parameter overlap, update magnitude, replay, delay, and
restart.<br>
**Novelty class:** N2/N2-T.<br>
**Known foundation:** continual-learning forgetting/transfer metrics, gradient
interference, and system-identification response analysis.<br>
**Irreducible delta:** a perturbation-to-behavior operator rather than one
post-task score.<br>
**Why this is not merely A + B:** it becomes more than N2 only if a
low-dimensional order parameter or phase boundary emerges.<br>
**Changed grammar or transfer mechanism:** retention is a dynamic susceptibility,
not a stored property.<br>
**New prediction:** matched immediate task-B performance can produce sharply
different A retention depending on overlap/update impulse, even with the same
replay budget.<br>
**Cheapest killing test:** a factorial two-task shared-MLP experiment with
controlled feature and gradient overlap.<br>
**Prior-art threats:** gradient-similarity analyses and continual-learning phase
diagrams.<br>
**Novelty confidence:** 20–40%; cutoff 2026-07-17.<br>
**Scientific value:** could turn “catastrophic forgetting” into a predictable
mechanism.<br>
**Publishable if successful:** empirical response law.<br>
**Publishable if partially successful:** benchmark and diagnostic protocol.<br>
**Publishable if it fails informatively:** evidence that no stable low-dimensional
law transfers across learners.

### Candidate E4 — Revocable experience

**Central claim:** deleting a corrupted or unauthorized experience and its derived
state can approximate clean retraining while preserving unrelated capabilities.<br>
**Novelty class:** N2.<br>
**Known foundation:** machine unlearning, data deletion, provenance, and
incremental view maintenance.<br>
**Irreducible delta:** propagate retraction across beliefs, updates, evaluations,
knowledge claims, and retention certificates.<br>
**Why this is not merely A + B:** likely A+B unless derived-agent-state
dependencies require a new minimal invalidation calculus.<br>
**Changed grammar or transfer mechanism:** experience is an admissible and
retractable premise, not permanent ground truth.<br>
**New prediction:** explicit dependency invalidation removes more harmful behavior
per unit retraining than replay-frequency heuristics.<br>
**Cheapest killing test:** inject one mislabeled experience group, retract it, and
compare with full clean retraining.<br>
**Prior-art threats:** exact/unlearning approximations, data lineage, and
truth-maintenance systems.<br>
**Novelty confidence:** 10–25%; cutoff 2026-07-17.<br>
**Scientific value:** safety and governance value even if scientifically
incremental.<br>
**Publishable if successful:** agent-state unlearning protocol.<br>
**Publishable if partially successful:** invalidation dependency benchmark.<br>
**Publishable if it fails informatively:** a lower bound showing full retraining is
unavoidable.

## 6. Transformational candidates

These candidates survived the subtraction and grammar tests only provisionally.
Their classifications should be downgraded immediately if the killing experiment
reduces them to ordinary provenance, safe policy improvement, confidence
intervals, or ensemble redundancy.

### Candidate T1 — Causal Update Certificate

**Central claim:** An agent update should be a quarantined candidate state plus a
machine-checkable statistical certificate; only verified causal improvement and
retention evidence may promote it to live state.<br>
**Novelty class:** provisional N3.<br>
**Known foundation:** ML provenance, event sourcing, safe policy improvement,
proof-carrying code/actions, preregistered evaluation, and checkpoint manifests.<br>
**Irreducible delta:** promotion depends on one certificate binding consumed
experience identities, before/after state, disjoint outcomes, counterfactual
controls, interference, and restored behavior.<br>
**Why this is not merely A + B:** ordinary provenance records what happened and
PCAA governs action authority; neither, in the searched sources, makes a
heterogeneous learning update's causal improvement and retention the admissibility
condition for state promotion.<br>
**Changed grammar or transfer mechanism:** `update(state)` is replaced by
`propose(state, evidence) -> candidate + certificate`, followed by an independent
`verify -> promote/reject`. Learning becomes a proof-obligated state transition.<br>
**New prediction:** mutation tests that preserve schema validity but swap lineage,
split, version, control, or restore identities will be accepted by typed logs and
rejected by the certificate verifier.<br>
**Cheapest killing test:** build certificates for finite and one small neural
trace; mutate each semantic dependency independently; compare false acceptance,
false rejection, localization, and overhead with the current records.
The **null hypothesis** is that ordinary typed provenance detects the same defects.
Success requires a materially larger semantic-defect class with low false
rejection; ordinary schema-equivalent performance is the strongest conventional
signature and triggers downgrade.<br>
**Prior-art threats:** [Proof-Carrying Agent Actions](https://arxiv.org/abs/2606.04104),
proof-carrying data, PROV-ML, model cards, Seldonian/safe-improvement methods,
event-sourced workflows, and reproducible ML pipelines.<br>
**Novelty confidence:** 20–40%; cutoff 2026-07-17; targeted academic and repository
search only.<br>
**Scientific value:** converts vague “the agent learned” claims into independently
falsifiable state-transition obligations.<br>
**Publishable if successful:** a new update semantics plus mutation benchmark.<br>
**Publishable if partially successful:** a semantic claim-linting system.<br>
**Publishable if it fails informatively:** evidence that ordinary provenance plus
safe evaluation is sufficient and Prospect should remain an integration harness.

### Candidate T2 — Trial-Governed Learning State Machine

**Central claim:** Repeated persistent agent updates can be governed as adaptive
candidate-versus-incumbent trials that control false promotion while preserving
old capabilities.<br>
**Novelty class:** provisional N3-T.<br>
**Known foundation:** adaptive clinical trials, sequential testing,
always-valid inference, HCOPE, safe policy improvement, canary deployment, and
shadow evaluation.<br>
**Irreducible delta:** every model, memory, harness, retrieval, calibration, or
policy update is the same kind of treatment candidate with external utility and
retention/noninferiority co-endpoints.<br>
**Why this is not merely A + B:** the proposed transfer changes the unit of agent
learning from an optimizer step to a governed experimental arm. It fails the
novelty test if ordinary SPI already expresses all heterogeneous updates and
dependent data correctly.<br>
**Changed grammar or transfer mechanism:** candidate state is randomized on a
shadow fork; preregistered sequential evidence, not training completion, triggers
promotion. The mechanism transfers randomization, interim analysis, and controlled
stopping from adaptive trials.<br>
**New prediction:** under repeated candidate generation, this state machine lowers
false promotion at the same accepted-gain rate relative to naive checkpoint
selection and fixed repeated tests.<br>
**Cheapest killing test:** a nonstationary two-task simulator with synthetic good,
neutral, and harmful candidate updates. Compare fixed tests, naive best-checkpoint
selection, native SPI/HCOPE, and the trial state machine. The **null hypothesis** is
no improvement in false-promotion control or sample efficiency after dependence is
accounted for. Decisive plots are false promotion versus accepted true gain and
episodes consumed. Abandon if it is ordinary SPI with renamed update types.<br>
**Prior-art threats:** the [FDA adaptive-design guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/adaptive-design-clinical-trials-drugs-and-biologics-guidance-industry),
group-sequential trials, bandits, safe policy iteration, deployment canaries, and
Seldonian optimization.<br>
**Novelty confidence:** 15–35%; cutoff 2026-07-17; bridge-field terminology is a
major uncertainty.<br>
**Scientific value:** gives continual self-modification an explicit error-control
problem.<br>
**Publishable if successful:** transfer theory and an update-promotion algorithm.<br>
**Publishable if partially successful:** an operational state machine with
calibrated limitations.<br>
**Publishable if it fails informatively:** proof that native safe policy
improvement already subsumes the formulation.

### Candidate T3 — Metrologically Traceable Knowledge Certificate

**Central claim:** Agent knowledge should be represented as a scoped capability
measurand with a traceability chain, uncertainty budget, version, validity domain,
and recalibration/expiry rule.<br>
**Novelty class:** provisional N3-T.<br>
**Known foundation:** measurement science, AI benchmark metrology, calibration,
hierarchical evaluation, and evidence-bearing knowledge claims.<br>
**Irreducible delta:** the object governs whether a learned capability may be
reused after environment, evaluator, model, calibration, or checkpoint changes.<br>
**Why this is not merely A + B:** a confidence interval attached to a benchmark is
not enough; the candidate changes “knowledge” from internal certainty to a
versioned measurement whose traceability and uncertainty determine runtime
validity. It is downgraded if that distinction never changes an action or claim.<br>
**Changed grammar or transfer mechanism:** measurand replaces latent belief as the
unit of knowledge; calibration and traceability become runtime operators, not
report metadata.<br>
**New prediction:** the certificate rejects apparent gains below evaluator
resolution or outside the measured domain even when model confidence and mean
benchmark score are high.<br>
**Cheapest killing test:** cross evaluator version, environment drift, seeds,
learner update, and restart; compare certificate decisions with a paired confidence
interval and a hierarchical model. The **null hypothesis** is that the object never
changes a valid promotion/reuse decision. Abandon if it is only a verbose interval.
Measure coverage, false scope extension, calibration drift, and decision impact.<br>
**Prior-art threats:** [Metrology for AI](https://arxiv.org/abs/1911.01875),
[JCGM GUM](https://www.iso.org/sites/JCGM/GUM-introduction.htm), NIST
metrological traceability, measurement assurance, benchmark datasheets, and
uncertainty-aware model cards.<br>
**Novelty confidence:** 20–45%; cutoff 2026-07-17; metrology/AI bridge work may use
different terminology.<br>
**Scientific value:** provides a rigorous definition of scoped, revisable
knowledge and exposes evaluator resolution as part of agent maturity.<br>
**Publishable if successful:** new runtime epistemology and measurement protocol.<br>
**Publishable if partially successful:** evidence schema plus benchmark
resolution diagnostics.<br>
**Publishable if it fails informatively:** demonstration that standard
hierarchical uncertainty already supplies every decision-relevant property.

### Candidate T4 — Error-Correcting Competence Code

**Central claim:** Redundant expressions of one capability across predictive
weights, episodic exemplars, and executable skills can support parity checks that
localize corruption and select minimal repair evidence.<br>
**Novelty class:** speculative N3-T.<br>
**Known foundation:** error-correcting codes, ensembles, modular continual
learning, consistency regularization, and checkpoint rollback.<br>
**Irreducible delta:** a learned competence syndrome maps cross-substrate
behavioral disagreement to a localized repair operation.<br>
**Why this is not merely A + B:** it is merely ensemble redundancy unless parity
structure identifies which substrate is faulty and repairs it more efficiently
than voting or rollback.<br>
**Changed grammar or transfer mechanism:** competence is a constrained codeword
across heterogeneous stores; disagreement is an observable syndrome, not just
uncertainty.<br>
**New prediction:** under single-substrate corruption, syndrome-guided repair
localizes the fault and restores behavior with less evidence while preserving
unrelated tasks.<br>
**Cheapest killing test:** encode one small capability in a predictor, exemplar
store, and skill; inject single and correlated corruption; compare voting,
rollback, replay, and syndrome repair. The **null hypothesis** is no localization
or sample-efficiency advantage. Abandon on correlated faults that make the
syndrome nonidentifiable or if voting matches it.<br>
**Prior-art threats:** coding-theoretic classifiers, co-training, multi-view
consistency, ensemble disagreement, modular redundancy, and fault-tolerant neural
systems.<br>
**Novelty confidence:** 10–20%; cutoff 2026-07-17; highest prior-art and
feasibility risk in the portfolio.<br>
**Scientific value:** a high-risk route from passive retention measurement to
active competence repair.<br>
**Publishable if successful:** a new fault model and repair mechanism.<br>
**Publishable if partially successful:** corruption-localization benchmark.<br>
**Publishable if it fails informatively:** evidence that agent competence lacks
the stable redundancy needed for code-like repair.

### Transformation-test summary

| Candidate | A-plus-B / subtraction result | Grammar change | Distinct prediction | Necessity/compression verdict |
|---|---|---|---|---|
| T1 | components known; residual is certificate-gated causal update promotion | mutation → proposal/verification/promotion | semantic mutations rejected beyond schema validation | provisional pass; dies if provenance + SPI is equivalent |
| T2 | trials + SPI known; residual is one governed state machine for heterogeneous self-modification | optimizer update → experimental arm | repeated false-promotion frontier improves | provisional pass; dies if native SPI handles dependence and all update types |
| T3 | metrology + evaluation known; residual is runtime-validity-bearing measurand | internal certainty → scoped capability measurement | evaluator resolution/traceability changes reuse decisions | provisional pass; dies if it reduces to a confidence interval |
| T4 | coding + ensembles known; residual is competence syndrome and repair | capability object → constrained multi-substrate codeword | fault localization and lower-evidence repair | weak pass; highest downgrade risk |

## 7. Cross-domain transfers

The transfers below reuse the corresponding idea cards; they are not additional
candidate counts.

| Donor → Prospect | Structural mapping | Preserved causal mechanism | Broken correspondences and required invention | Adoption barrier / enabling change | Recipient-specific prediction | Audit status |
|---|---|---|---|---|---|---|
| Database WAL/event sourcing → lifecycle recovery | database state→agent state; transaction→experience/update; log record→typed lifecycle event; commit/replay→promotion/recovery; prefix consistency→no partial learning | durable ordered intent plus idempotent replay prevents partial commits | learners may be nondeterministic; environments have irreversible side effects; state payloads and schemas evolve; compensation may replace rollback | needs durable journal, idempotency keys, environment reconciliation, and transactional learner API | crash at every boundary never duplicates experience or silently applies an update twice | known/N1 diagnostic transfer; [The Log is the Agent](https://arxiv.org/abs/2605.21997) is close |
| Metrology → knowledge certificate | measurand→capability; instrument reading→evaluation episode; calibration→evaluator audit; traceability chain→claim lineage; uncertainty budget→combined evaluation/learner/restore uncertainty | a measurement is meaningful only with scope, calibration, traceability, and resolution | learning changes the measurand; evaluator and agent can co-adapt; errors are dependent/non-Gaussian; no immutable reference standard | needs versioned evaluators, repeated measurements, uncertainty decomposition, and runtime validity rules | some apparent gains become “below resolution” or expire after evaluator drift | **rare, measurement/diagnostic, provisional N3-T** |
| Adaptive clinical trials → update promotion | treatment arm→candidate state; patient allocation→shadow episode routing; outcome→external utility; adverse event→capability regression; stopping rule→promotion | randomization and controlled interim analysis limit biased repeated selection | episodes are dependent; policies change future data; outcomes are delayed/multiobjective; withholding a better policy has cost | needs safe shadowing, always-valid inference, dependence-aware design, and retention co-endpoints | false promotion falls without an equal loss of accepted genuine gains | **rare, provisional N3-T** |
| Proof-carrying code/actions → causal update certificate | program/action→candidate update; proof obligation→causal evidence requirements; checker→independent verifier; admission→promotion | untrusted producer supplies an artifact accepted only by a small verifier | improvement claims are statistical; environment is incomplete; verifier/evaluator can be gamed; proof may expire under drift | needs canonical claim language, counterfactual controls, mutation suite, and trust boundary | schema-valid lineage/split/version attacks fail verification | direct adjacency, provisional N3/N3-T |
| Error-correcting codes → competence repair | codeword→multi-substrate competence; parity check→behavioral invariant; syndrome→fault localization; decode→repair | structured redundancy makes some corruptions detectable and correctable | faults are continuous/correlated; no fixed codebook; behavioral probes are costly; repair can alter all parameters | needs deliberately redundant representations, stable parity probes, and sparse fault assumptions | single-substrate corruption is localized and repaired with less evidence | **rare, high-risk provisional N3-T** |
| Causal inference/experimental design → experience attribution | treatment→experience group; potential outcome→behavior under deletion/reweighting; randomization→controlled experience assignment; effect→held-out behavior delta | counterfactual contrasts separate association from causal contribution | online policies select their own data; interference occurs among experiences; exact retraining is expensive; positivity can fail | needs randomized or instrumented collection and exact small-model ground truth | replay frequency and causal contribution systematically diverge | N2-T unless a new longitudinal operator emerges |

### Transfer validation

- **Terminology-removal:** each row remains expressible as state, observation,
  operator, invariant, noise, and failure mode without donor vocabulary.
- **Structural/homomorphism:** the crucial donor operation maps to an admissible
  Prospect operation, but none is exact; the broken correspondences above are the
  research burden.
- **Causal preservation:** ordered durable intent, randomization, traceability,
  proof checking, redundancy, and counterfactual contrast retain their original
  causal roles.
- **Counter-analogy:** nondeterministic learning, policy-dependent data, and
  environment side effects are shared major mismatches. Any one may be fatal.
- **Native baseline:** compare respectively with workflow recovery, SPI/HCOPE,
  paired/hierarchical evaluation, typed provenance, ensemble voting/rollback, and
  influence/data-valuation methods.
- **Historical obviousness:** all donor mechanisms are old enough to have been
  suggested earlier. What changed is the emergence of long-running,
  heterogeneous, self-modifying agent harnesses and auditable event substrates;
  this supports feasibility, not automatic novelty.

## 8. New-evidence discovery programs

These programs can create observations absent from the current corpus. They become
N4/N4-T only if an experiment produces a new phenomenon that materially induces a
new hypothesis.

### Program P1 — Causal Chain Mutation Laboratory

**What is varied:** future leakage, split contamination, task-ID leakage, omitted
or fabricated ancestry, false versions, marginal-preserving and
marginal-changing shuffles, duplicate replay, missing checkpoint components,
restored-process misattribution, and swapped control labels.<br>
**What is measured:** false acceptance, false rejection, defect localization,
verification latency, artifact size, and evaluator agreement for typed logs versus
T1 certificates.<br>
**Surprising outcome that matters:** a small certificate language catches a broad
class of semantic false claims that remain schema-valid, across two unrelated
learner backends.<br>
**Hypothesis induced:** causal update verification has a reusable defect algebra,
not just application-specific checks.<br>
**Leakage/bug exclusion:** generated traces have a ground-truth dependency DAG;
mutations are single-factor first, then composed; verifier authors do not see the
held-out mutation family; independent oracle replay adjudicates disagreements.<br>
**Null hypothesis:** certificates add overhead but no semantic detection beyond
ordinary typed provenance.

### Program P2 — Interference Spectroscopy

**What is varied:** task feature/gradient overlap, shared versus isolated
parameters, update magnitude/order, replay capacity and sampling, delay, restart,
and update type (weights, memory, retrieval, harness).<br>
**What is measured:** predictive proper scores, realized utility/regret,
calibration, plasticity, backward/forward transfer, behavioral drift, state drift,
and restart parity.<br>
**Surprising outcome that matters:** retention collapses along a reproducible
low-dimensional boundary or susceptibility measure across learners.<br>
**Hypothesis induced:** retainability is predictable from an interference response
operator rather than model family labels.<br>
**Leakage/bug exclusion:** opaque task identities, frozen factorial cells,
independent seeds, shared evaluation streams, parameter-overlap verification,
isolated-task negative controls, and fresh-process checks.<br>
**Null hypothesis:** no transferable response law remains after model/task effects.

### Program P3 — VOI Sign-Reversal Atlas

**What is varied:** likelihood misspecification, sensor reliability, information
cost, downstream goal structure, distribution shift, and irrelevant high-entropy
variables.<br>
**What is measured:** predicted EIG/EVSI, realized posterior change, external
utility, regret, calibration, and sign error.<br>
**Surprising outcome that matters:** stable regimes where predicted information
gain is positive and accurate about belief change yet realized decision value is
negative.<br>
**Hypothesis induced:** epistemic estimators require a two-stage calibration from
belief change to decision consequence.<br>
**Leakage/bug exclusion:** exact finite oracles, label permutation invariance,
counterfactual replay, cost-matched controls, hidden test worlds, and a second
implementation of the scorer.<br>
**Null hypothesis:** standard EVSI with correct costs and likelihood calibration
eliminates every apparent sign reversal.

## 9. Prior-art and scientific-value audit

### Adversarial threat matrix

| Candidate claim | Strongest prior-art threat | Facet-level overlap | Verdict under current search |
|---|---|---|---|
| agent stores/reuses experience | ExpeL, Reflexion, Voyager, ExpWeaver | direct mechanism and outcome overlap | likely known |
| harness learns from executions | MemoHarness | direct problem and mechanism overlap | likely known |
| reflector learns cross-task adaptation | Training Language Agents to Learn from Experience | direct learned-adaptation overlap | likely known |
| actions balance utility and information | VOI, Bayesian design, active inference | direct objective overlap | likely known |
| replay limits forgetting | Continual World/Continual-Dreamer and continual learning | direct mechanism and metrics overlap | likely known |
| event history is authoritative/replayable | event sourcing and The Log is the Agent | direct architecture overlap | likely known |
| updates/actions carry evidence | PCAA, proof-carrying data/code, PROV-ML | certificate and verifier adjacency | known components, possibly new relationship |
| candidate policies require safe evidence | SPI/HCOPE/Seldonian/adaptive trials | promotion and confidence overlap | known components, possibly new heterogeneous-update formulation |
| benchmark results carry traceability/uncertainty | metrology for AI, GUM, measurement assurance | direct measurement concepts | known components, possibly new runtime-validity role |
| one object binds causal ancestry, disjoint improvement, shared interference, and restart | no exact match found in the targeted search | closest work covers subsets | apparently sparse; insufficient evidence for a strong novelty claim |

### Pareto frontier

Scores are 0–5. For **first-test cost**, 5 means cheapest.

| Candidate | Apparent novelty | Falsifiability | Importance | Feasibility | First-test cost | Informative failure | Publication potential |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 same-chain learner | 1 | 5 | 5 | 5 | 5 | 5 | 3 |
| E1 VOI scorecard/atlas | 2 | 5 | 4 | 4 | 4 | 5 | 4 |
| E3 interference spectroscopy | 2 | 5 | 5 | 4 | 3 | 5 | 4 |
| T1 causal update certificate | 3 | 5 | 5 | 3 | 3 | 5 | 4 |
| T2 trial-governed learner | 4 | 4 | 5 | 2 | 2 | 5 | 5 |
| T3 knowledge certificate | 3 | 5 | 4 | 4 | 4 | 5 | 4 |
| T4 competence code | 4 | 4 | 4 | 2 | 2 | 4 | 4 |

The Pareto set is:

- **foundation and fastest validation:** R1 same-chain learner;
- **best measurement direction:** T3 knowledge certificate;
- **best systems hypothesis:** T1 causal update certificate;
- **best mechanism study:** E3 interference spectroscopy;
- **high-risk formulation change:** T2 trial-governed learner;
- **cheap informative-negative direction:** E1/P3 VOI sign-reversal atlas; and
- **strange but valid reserve:** T4 competence code.

## 10. Recommended first experiment

Build one small real shared-parameter learner inside the authoritative runtime
before pursuing a large model, external arena, or transformational mechanism.

### Frozen protocol

1. Generate opaque task-A and task-B identities with controlled feature/gradient
   overlap.
2. Freeze train, calibration, predictive-held-out, behavioral-held-out, and
   retention streams; seeds; budgets; metrics; and pass thresholds.
3. Evaluate task A before learning.
4. Let the authoritative runtime choose and collect A experiences.
5. Train only from those exact transition identities and issue a
   version-changing `UpdateReceipt`.
6. Compare true-link, no-update, irrelevant-evidence, and
   marginal-preserving-link-permutation arms on disjoint prediction data.
7. Execute frozen pre/post policies on identical held-out outcomes at equal
   environment/model/planner budget.
8. Learn task B through the same shared parameters and re-evaluate A and B.
9. Save every stateful category, start a fresh process, restore, and re-evaluate A
   and B.
10. Run the first causal-chain mutations against the resulting trace.

Measure log loss/Brier score, calibration, realized utility/regret, plasticity,
forgetting, restored behavioral parity, consumed-ID agreement, and mutation false
acceptance. Preserve every raw arm whether it passes or fails.

**Null hypothesis:** correct linked experience produces no disjoint predictive or
behavioral improvement beyond the matched controls, or any gain does not survive
genuine interference and fresh-process restore.<br>
**Decisive observation:** the correct-link arm alone changes persistent model
state, improves disjoint prediction and executed utility, retains a preregistered
fraction of the A gain after B and restart, learns B, and rejects injected false
lineage claims.<br>
**Cost:** approximately two engineering days for a minimal shared MLP/task
generator after the current runtime contracts; CPU-scale formal runs.<br>
**Abandonment criterion:** if the ordinary learner cannot pass the causal lane
without ad hoc task leakage or isolated task state, stop architecture expansion
and fix the learning/measurement loop. If it passes but certificates add no
detection beyond typed records, retain Prospect as an auditable harness and
downgrade T1.

This experiment is intentionally smaller than an external arena. It resolves
whether Prospect has a learning architecture to benchmark before benchmark
variance, model scale, and infrastructure cost obscure the answer.

## 11. Audit limitations

- The literature search was targeted, not exhaustive. Patents, dissertations,
  non-English work, closed industrial systems, and terminology from formal
  verification or biostatistics may contain closer matches.
- MemoHarness, PCAA, and The Log is the Agent are 2026 preprints and too recent
  for mature replication or citation structure; they nevertheless invalidate
  broad novelty language now.
- No candidate above has experimental N4/N4-T evidence.
- The portfolio does not select a production model architecture. That is
  intentional: a model choice cannot repair a disconnected causal evaluation.
- Publication potential scores are relative research judgments, not promises.
- The current Prospect implementation establishes contracts and narrow
  diagnostics only. Any novelty classification must be re-audited after the
  first real learner produces evidence and before a public claim enters the
  README, roadmap, or ARA claims.

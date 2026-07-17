# MM-009 causal deformation/appearance prediction protocol

**Date:** 2026-07-16  
**Status:** pre-real candidate. It becomes design-frozen only when the pre-real
scientific audit records GO against the exact protocol/config/source hashes and an
exclusive `freeze-record.json` is durable. No MM-009 real prediction or score may be
computed before that event.  
**Parent:** sealed MM-007 (`physically_matched_resolution_failure_supported`)  
**Supersedes:** the untouched real-target route of the MM-008 v2.2 preseal candidate;
MM-008's development evidence and exact finite-grid mathematical sources remain
available as dependencies, but no MM-008 reserved/challenge seed or real-target run is
licensed or reclassified.

## 1. Question and claim boundary

MM-009 asks whether an operator estimated only from the observed half-second change
`previous -> current` can predict the next half-second change when that same frozen
operator is applied once to `current`:

```text
fit K(previous, current)
freeze K and K(current)
score K(current) against future
```

The three independently preregistered families are a finite exact-global affine grid,
global per-channel appearance, and their spatial-then-appearance combination. Each
family has its own gate. There is no per-row, per-video, or outcome-selected best-arm
envelope. Subject to the global precedence in section 9, an overall GO requires at
least one family to pass its complete frozen gate. An invalid or inconclusive state in
**any** family globally preempts GO even if another family would otherwise pass. When
all families are decision-complete, every passing family, rather than an outcome-ranked
winner, is carried to the next bridge. Multiple passes without frozen conditional
dominance are classified as mechanism-nonidentifiable.

The eight videos and 453 transitions are already outcome-visible development data.
Videos, not rows or pixels, are the support units. A GO is an exploratory mechanism
selection result on this panel. It is not independent confirmation, population
prevalence, a deployable predictor, a learned TAESD dynamics result, an end-to-end
Prospect capability, multimodal fusion, planning, or control. A clean failure is
bounded to the exact tested operator, normalization, half-second lag, and pixel
representation. MM-001 through MM-007 remain sealed and unchanged.

## 2. Immutable parents and source closure

Before the formal marker, preparation may inspect MM-007 only as opaque filesystem
objects. Its top-level census requires exactly the eight regular non-symlink files
below plus one real non-symlink directory named `inputs/`. That directory is
lineage-only and non-authoritative: preparation checks only its top-level name and
type, never traverses it, never reads or hashes anything below it, and never copies it.
Any other top-level file, directory, or symlink is invalid. Preparation checks the
following live file modes and SHA-256 values and makes immutable `0444` copies with
the same bytes. After opaque hashing it repeats the top-level census: membership and
the root and pinned-file identities must be unchanged, and `inputs/` must retain the
same device, inode, and directory type. It must not parse JSON, decode the frame NPZ,
invoke an MM-007 scientific verifier, or construct a real row at that stage. The
authoritative files are:

```text
artifact-manifest.json     db0b6654ab098dc9a3ec93e4a6de8820bbe5860d44974645e9a5ee7dad1537fb  0644
input-manifest.json        1f83c805e6c5d75f4f1d5a2102d471c15bbc6bb787960cb5ae630bd2260faa1f  0644
formal-start.json          ea5c7bda870d71ead3172c1fc6e504d6a6b02d2ba785e9fd2fc75a91c667eee3  0444
MM-007-evidence.json       13dfa89e541e6122263ea9814d42fb328da303dcc74556cdaaa5d5860d99abaf  0644
MM-007-results.json        3c92729e1e5c18c14461e36602bdb86acd31750d9f5a85f535cd33a43fb9c47b  0644
MM-007-report.md           b18760128941ab2eff893b8c0afc469b92f71077d489e060d56519407990b8a2  0644
MM-007-protocol.md         24bbac1855cc2b51d2a65012b9c63037637c53555b86bbad7c66a6249108a73c  0644
MM-007-frames-64x64.npz    fbc79d81a06720175139f7106745bd58f8788f43cc5a2fcd10658d186909797f  0644
```

The frame archive must contain exactly:

```text
video_ids     06e75502f8c9ab7883ba6a44d9e0f250bd5f678ac8b5989b2b7b5349b69e4c50
timestamps    128c725db3361bf55c89017c02a4bd08f54622f09018d10c4c83b4467c4d3d55
frames_uint8  46d21d8c5b7d3a88abd96500ab07c3d54606a8f74b1500ddedeefb45e2d13eb9
```

The reused MM-008 v2.2 protocol identity is
`300a4e14bd0182b8ce9a9448d7b8261c51e20d67285a0988a33114a66bdb9622`.
The freeze record pins exact hashes of the prediction fit core (`geometry_v22.py`,
`fitting_v22.py`, `global_v22.py`, and `nongrid_v22.py`) and, separately, the formal
fixture dependencies (`calibration_v22.py` and `synthetic_v22.py`). Their certificates
authenticate historical fits only. MM-009 issues separate apply and prediction
identities and never calls an inherited reconstruction a forecast. The two MM-007
verifier entry sources are also in MM-009's reviewed source closure; that verifier in
turn recomputes the sealed MM-007 68-file source manifest before its result can be
accepted.

Preparation also records that the canonical MM-008 result root, formal v2.2 review
records, and verification records are absent. A supersession record states that the
preseal candidate was retired before reserved/challenge/real use. If that statement is
false at freeze time, preparation fails closed.

Only after the exclusive formal marker and the formal MM-009 synthetic gate are
durable may the existing MM-007 fast verifier run against the live pinned root. It must
pass before the copied evidence or frame archive is parsed and before real row or
normalizer construction. A verifier, tree, pin, or copied-byte mismatch terminates the
one-shot output as `invalid_MM009`; it cannot be repaired or retried in place.

## 3. Rows, folds, and source-only normalization

The archive has 477 rows ordered by `(video_id,timestamp)`. For each video and each
position `q` in `range(1, N-2)`, construct exactly one row:

```text
previous = frame[q-1]  at t-0.5
current  = frame[q]    at t
future   = frame[q+1]  at t+0.5
```

This yields exactly 453 rows and counts
`60,61,56,62,62,45,63,44` in lexicographic video order, with current timestamps
`1.5 + 0.5*i`. The canonical identity must equal MM-005's
`d4f87867c718370cd925c8dc2a4b01cc89ff4d18f52e9d309f53b5e81e0c8f3b`.
The four fixed folds are the lexicographic two-video test folds from MM-001.

Frames are converted to channel-first float values in `[0,1]`. For each fold, the
normalizer is fit only on the R8 pooled **current** frames belonging to the six
training videos:

```text
mean[c]  = mean(current_R8_train[:,c,:,:])
scale[c] = max(std(current_R8_train[:,c,:,:]), 1e-6)
```

That same three-channel mean and scale normalize R64 previous, current, future, and
source controls for the two held-out videos. Normalizer values and fingerprints must
exactly replay the corresponding sealed MM-007 R8 evidence. Future is never used to
fit or select a normalizer.

## 4. Frozen estimators and one-step application

All scientific arrays are C-contiguous little-endian float64. Central sites are the
2,304 row-major coordinates in `[8:56,8:56]`; every score has three channels and
6,912 scalar elements per row. The exact v2.2 full mask and finite 15,625-state grid
are used without modification.

For each row, fit full central `previous -> current` independently:

```text
affine:
    theta = exact_global_affine(previous, current)
    prediction = sample(current, theta)

appearance:
    gain,bias = two_pass_appearance(previous, current)
    prediction = gain * identity_sample(current) + bias

combined:
    theta,gain,bias = exact_global_combined(previous, current)
    prediction = gain * sample(current, theta) + bias
```

The v2.2 affine and combined requests share the same enumerated source stream.
Application is spatial first, appearance second. The selected operator is applied
exactly once to observed current. It is never refit, inverted, negated, doubled,
composed algebraically, clipped, or chosen using future error. In particular,
`GlobalResult.prediction` and `AppearanceEstimate.prediction` are reconstructions of
current from previous and cannot be saved as future predictions.

Each MM-009 apply record binds the historical fit identity, selected state and
parameters, selected gains/biases/retained IDs where applicable, current-array hash,
fresh prediction-array hash, shape, dtype, and a scalar replay. Fit scope contains
only previous/current/config bytes. Apply scope contains only the fit identity and
current bytes. Score scope is the first scope allowed to contain future bytes.

## 5. Historical controls and causal baselines

The primary future forecast uses the full historical fit. Historical
identifiability is tested separately by checkerboard cross-fitting: fit parity 1 and
predict parity 0, then fit parity 0 and predict parity 1. The two held-parity outputs
are joined in canonical site order. No full-fit training residual can satisfy this
control.

For every family and row retain:

- `history_identity`: previous used directly to reconstruct current;
- `history_xfit`: ordered checkerboard reconstruction of current from previous;
- `history_shuffle_xfit`: the same reconstruction after replacing previous with the
  fixed within-video far half-cycle row, while current remains fixed;
- `forecast`: the ordered full fit applied to current;
- `forecast_shuffle`: fit shuffled previous to current, then apply that frozen fit to
  actual current;
- `forecast_reverse`: fit current to previous, then apply that reverse-direction fit
  to current (never algebraically invert parameters);
- `persistence`: current predicts future;
- `velocity`: `2*current - previous` predicts future, with no clipping;
- `future_derangement`: score the ordered frozen forecast against the fixed
  within-video far half-cycle future, without refitting anything.

The far mapping for a video with `n` rows is `(i + ceil(n/2)) mod n`; if it has a
fixed point, preparation is invalid. Mapping, inverse membership, and row identities
are frozen before scoring.

All three families share the same matched, spatial-structure-erased, current-only
bias baselines. Historical bias is fit on the opposite-parity current cells and
broadcast to held parity. Forecast bias is fit on all observed current central cells
and broadcasts its frozen channel values to the future output sites. The corresponding
`u`, `b`, and `bd` error primitives must be bit-identical across affine, appearance,
and combined records for a video. These controls contain current-frame values but no
source spatial structure; they are scoring comparators and never enter an operator fit
or application. Constant/stationary denominators cannot yield source-use credit.

## 6. Aggregation and exact predicates

All errors are untrimmed SSE. Aggregate SSE and scalar count across every row of one
video before dividing to MSE. No row weighting, trimming, rounding, epsilon division,
or pooled-video test is allowed. Comparisons use multiplication on finite nonnegative
SSE values with identical positive counts.

For family `m` and video `v`, define:

```text
i = history_identity_mse
a = history_xfit_mse
q = history_shuffle_xfit_mse
p = persistence_future_mse
c = ordered_forecast_future_mse
h = shuffled_history_forecast_future_mse
r = reverse_fit_forecast_future_mse
z = value_velocity_future_mse
d = future_derangement_ordered_mse
pd = persistence_mse_against_the_same_deranged_future
```

For every family, additionally define `u=history_bias_mse`,
`b=forecast_bias_mse`, and `bd=forecast_bias_mse_against_the_same_deranged_future`.
A video has sufficient activity only when both `i` and `p` are strictly greater than
`1e-4`.

Historical support is:

```text
activity
and 1.25*a <= i
and 1.10*a <= q
and u > 0
and 1.25*a <= u
```

Future support is:

```text
activity
and 1.25*c <= p
and 1.10*c <= h
and 1.10*c <= r
and 1.10*c <= z
and 1.10*c <= d
and b > 0
and 1.25*c <= b
```

Joint support requires historical and future support on the **same video**. A family
passes only if all of the following hold:

1. activity holds on all 8/8 videos;
2. joint support holds on at least 6/8 videos;
3. the single same-video directional predicate
   `a<i and a<u and c<p and c<b` holds on at least 7/8 videos;
4. every two-video fold contributes at least one joint-supporting video;
5. every row, exact-global certificate, apply replay, hash, count, and finite check is
   complete;
6. fewer than three videos trigger that family's range warning; and
7. every family remains clear of the global null, integrity, isolation, activity, and
   mixed-state preemptions in section 9.

Affine/appearance passes do not require combined to pass. Combined is called a
joint-operator result only if combined itself passes, each single-family forecast SSE
is at least `1.10` times combined forecast SSE on at least 6/8 videos
(`1.10*c_combined <= c_affine` and `1.10*c_combined <= c_appearance` on each counted
video), those margin videos cover all four folds, and combined is strictly below both
single-family errors on at least 7/8 videos. This is an exact multiplicative relation,
not a claim of a 10% reduction under an alternative denominator. Otherwise multiple
passing families are
`causal_family_nonidentifiable`.

A range warning fires for a video when at least 25% of its rows select an applicable
bound: affine translation `ty/tx` at `+/-8`, affine gradient
`ayy/ayx/axy/axx` at `+/-4`, gain at `-2/+4`, or bias at `-4/+4`.
Nonfinite predictions, invalid Jacobians, out-of-bounds samples, or invalid selected
states invalidate the package immediately rather than merely warning for one family.

The four real null-support predicates are exact and independent of the prose for the
ordered gate:

```text
history_shuffle_null =
    activity and 1.25*q <= i and u > 0 and 1.25*q <= u

forecast_shuffle_null =
    activity and 1.25*h <= p
    and 1.10*h <= r and 1.10*h <= z and 1.10*h <= d
    and b > 0 and 1.25*h <= b

forecast_reverse_null =
    activity and 1.25*r <= p
    and 1.10*r <= h and 1.10*r <= z and 1.10*r <= d
    and b > 0 and 1.25*r <= b

future_derangement_null =
    pd > 1e-4
    and 1.25*d <= pd and bd > 0 and 1.25*d <= bd
```

Here every `1e-4` condition is MSE, equivalently SSE strictly greater than
`1e-4*count`; all displayed error comparisons use equal scalar counts. If any one
null reaches 6/8 in any family, the global result is
`invalid_MM009_real_negative_control`. A count of 3--5/8 globally produces
`MM009_inconclusive` with a family/null-specific reason. There is no separate
`inconclusive_MM009_real_negative_control` label. Null counts of 0--2 do not preempt.

Because eight outcome-visible videos do not justify an independence model, `6/8` is
an exploratory branch threshold, not a p-value. With every denominator, control,
certificate, and range check valid, future and joint support both at most 2/8 form a
clean bounded family NO-GO only when historical support is also at most 2/8
(`tested_family_identifiability_failure`) or at least 6/8
(`historically_identifiable_but_complete_future_gate_failed`). This is a bounded
compound-gate diagnosis, not direct evidence of nonstationarity. Historical or future support at
3--5, future support at least 6 without joint support, activity loss, or mixed control
behavior is inconclusive.

## 7. Synthetic and mutation gates

The formal MM-009 synthetic panel uses only these exposed seeds:

```text
base positive/negative seeds: 990900, 990901, 990902
independent-future seeds:     990910, 990911, 990912
channel-permutation seed:     990919
```

The base seeds are applied to repeated translation, affine, appearance, and combined
positives. Each positive recovers its declared applicable operator, performs a
bit-exact fresh apply replay, and satisfies the complete historical, future, joint,
and same-fixture directional predicates using its ordered, deterministic row-3
shuffled, reverse-fit, velocity, deranged-future, and matched-bias comparators. The
same base seeds also define:

- stationary fixtures, which require persistence SSE exactly zero, every arm SSE at
  most `1e-24`, and ineligibility;
- affine and appearance reversal checks, which require the repeated-direction
  forecast not to cross the `1.25` persistence margin when the previous frame is used
  as the future;
- constant-target fixtures, in which every arm matches the current-only bias forecast
  within SSE `1e-22`; and
- coupled-boundary fixtures, in which affine and combined each select at least one
  applicable registered boundary.

Independent fixtures pair `(990900,990910)`, `(990901,990911)`, and
`(990902,990912)`. Across the three-fixture aggregate, every family must lack joint
source-use credit: historical cross-fit cannot beat both identity and matched bias by
the `1.25` factor, and future forecast cannot beat both persistence and matched bias
by that factor. The declared branch is `tested_family_identifiability_failure`.
`PCG64(990919)` separately requires exact channel-permutation equivariance of one
apply. All seed identities and the configuration/protocol bindings are recorded. The
MM-008 reserved seeds `820800..820807` and both retired nonce receipts are forbidden.

Exact-grid enumeration, candidate/tie behavior, checkerboard parities, transpose
semantics, and source/parameter/prediction forgery rejection are covered by inherited
v2.2 and focused MM-009 unit/mutation tests before freeze; those tests are not formal
MM-009 synthetic evidence. No acceleration or periodic-ambiguity scenario is part of
the formal MM-009 control panel.

After the formal control record is written, a refit-free validator requires its
exact schema and census and independently replays every stored predicate, margin,
aggregate independent-future branch, seed, and digest before parent access. Its
validated durable canonical bytes must then be bit-identical to the trusted in-memory
control result. A self-consistent replacement record with a recomputed digest cannot
authorize the run.
The semantic package verifier reruns the complete panel and requires bit-identical
evidence.

After all 453 predictions are frozen and before any score process starts, one fresh
`-I -S -B` fitting-free process runs the complete future-isolation gate. It imports
custody/record code only and rejects the predictor, worker, and all four v2.2 fitting
modules if present. It snapshots the prediction freeze plus the three canonical files
for each row (`1 + 3*453 = 1360` files) and all 453 detached target files. For each row
ordinal `o`, it validates in memory exactly:

```text
random:   PCG64(991000 + o).standard_normal((3,64,64), dtype=float64)
reverse:  future[:, ::-1, ::-1]
derange:  that row's preregistered deranged_future
byte:     XOR uint64 LSB at [0,8,8]
NaN:      set [0,8,8] to NaN and require target validation to reject
```

The four finite variants must each validate for all 453 rows (1,812 validations); all
453 NaN variants must reject. Mutations are never written. Prediction-side and target
file censuses, modes, bytes, and aggregate manifest hashes must be identical before
and after the sweep. A fitting-module import, mutation-census mismatch, file change,
or source-side identity change is `invalid_MM009_future_isolation` and globally
preempts interpretation.

## 8. Custody, sandbox, and prediction freeze

Preparation may opaque-hash/copy parent files and run synthetic-only tests. It may not
decode the real frame NPZ, parse real scientific evidence, construct a real row, or
invoke a real predictor before the exclusive formal marker and formal synthetic gate.
After those records are durable, the existing MM-007 fast verifier must first pass as
specified in section 2. Only then does one custodian parse the copied parent,
reconstruct rows and normalizers, and write detached row inputs. Each primary row
input contains only its identity, fold normalizer, previous, current, and
preregistered source-control arrays. Its future is written to a separate target
location.

One fresh predictor process is launched per row through an MM-009-owned Linux
Landlock/seccomp launcher under CPython flags `-I -S -B`. Preparation makes and
manifests two immutable copied dependency closures: only `numpy/` plus `numpy.libs/`,
and the Python standard library with `site-packages`, `__pycache__`, and bytecode
caches excluded. Package-manager source hard links are admitted only as separately
bound path/size/SHA byte sources and are dereferenced into unique `0444` destination
files. All closure directories are `0555`. Live venv, Conda, and system Python package
trees are absent from the post-bootstrap `sys.path` and receive no Landlock rule.

CPython startup and the fixed standard-library-only launcher bootstrap necessarily
precede path rebinding and Landlock. The exact interpreter, Python/cache-tag identity,
and live bootstrap standard-library root are therefore recorded as an explicit trusted
host-bootstrap boundary. After rebinding, only the copied standard library is an
import root. Read-only `/usr/lib/x86_64-linux-gnu`, `/usr/lib64`, the loader cache,
`/dev/null`, and CPU-feature paths form the narrower post-sandbox host ABI/operational
boundary; they are not presented as MM-009 scientific source. Scientific authority is
limited to the copied shadow package containing the exact four v2.2 mathematical
modules and MM-009 source-only predictor code, copied dependencies, config, and that
one source row. A separate copied custody runtime contains only record, preparation,
scoring, score-worker, future-isolation, launcher, and sandbox code plus the two
custody launchers; predictor, worker, and all v2.2 mathematical modules are physically
absent and unimportable.

This host reports Landlock ABI 6. Before importing NumPy or task-specific worker code,
every predictor, score, and future-isolation launcher closes every descriptor above
stderr, sets `PR_SET_NO_NEW_PRIVS`, installs the libseccomp filter, and installs a
deny-by-default Landlock ruleset handling all ABI-3 filesystem rights plus network
bind/connect. Only one fresh output directory receives write/create/truncate rights.
The libseccomp allow-default filter returns `EACCES` for process replacement and
process-group escape, all socket/socketpair/connect/bind/listen/accept/send/receive
syscalls, and ptrace, cross-process memory, namespace/mount, keyring, BPF, and
descriptor-theft syscalls. Before the formal marker, a known-answer probe must demonstrate denial of
the live repository, parent frame archive, an unlisted sentinel, cross-process reads,
and all six `(AF_INET, AF_INET6, AF_UNIX) x (SOCK_STREAM, SOCK_DGRAM)` socket-creation
variants. It additionally proves denial of every registered live Python root,
including venv/Conda/system package directories. After real rows are detached but
before the first predictor, the probe is repeated against an actual sibling source row
and every target-row directory; any readable path or available network variant
terminates the formal run.

Each child starts with a fixed empty working directory, fixed argv, umask `077`, one
thread, a minimal allowlisted environment, and no inherited file descriptors. Empty
shadow namespace-package initializers prevent broad `bench` imports. Scorer and
lifecycle code are absent from the sealed runtime. The child uses no RNG and exits
after one row. This Landlock/seccomp contract replaces an earlier pre-freeze
`bubblewrap` draft after a host probe showed unprivileged user-namespace UID mapping
is disabled; no real data had been opened and no protocol freeze existed.

Each child writes exactly two temporary `0444` files in its private directory:
`predictions.npy` and `prediction.json`. The supervisor validates the exit, exact
source/control row bindings, modes, file hashes, roles, fit/apply records, parameter
reapplications, nested global/non-grid certificates and contexts, parity records,
aggregate evidence digest, and prediction bytes before it copies the two files as canonical
`predictions.npy` and `worker-evidence.json`. It then exclusively writes a separate
canonical `commit.json` binding row/fold/video identity, source file, both copied
worker files, config/protocol, and the predecessor supervisor-commit hash. Exclusive
creation, fsync, `0444` mode, and directory fsync apply to the canonical files. Child
JSON is evidence and never masquerades as the supervisor chain. Interrupted,
duplicate, reordered, or partial rows terminate the run; there is no in-place resume
or retry. The child is revalidated immediately before copying, and the canonical pair
is revalidated before its supervisor commit. On the first failure or deadline, a
shared supervisor cancellation state terminates and then kills every registered child
process group; no commit, target access, or continuation is permitted while a child is
registered. Only after all 453 supervisor commits are durable and
`prediction-freeze.json` binds their complete chain may any process open a target
row. No predictor remains alive then.

One fresh `-I -S -B` score-only process is then launched per row. Its Landlock rules
admit only that row's source, target, prediction, supervisor commit, worker evidence,
configuration, copied import/ABI roots, and empty score directory; sibling targets and
the live repository remain unreadable even through a raw open or manual source
loader. It has no fitting or predictor API, validates all row-local identities, loads
that one detached future, records only the 13 primitive SSE/count pairs per family,
writes exactly one immutable `score.json`, and exits. Each score record binds the
supervisor commit, prediction, source, and target hashes. Future bytes never enter a
fit, apply, prediction, worker-evidence, or supervisor-commit hash.

The future-isolation child similarly receives only the prediction freeze, frozen
prediction tree, complete detached-target tree, copied import/ABI roots, and a fresh
empty private output directory. The supervisor validates its exact evidence and
current source/target snapshots, copies it exclusively to the canonical destination,
revalidates it, and removes the private directory before scoring.

Immediately after `prediction-freeze.json` and before the future-isolation process or
any scorer can open a target, the supervisor requires the future, score, evidence,
result, report, manifest, and complete score tree to be wholly absent. It snapshots
every current regular artifact by path/mode/bytes/SHA-256 and writes immutable
`pre-score-budget.json`. The projection is

```text
current artifact bytes
+ 1,000,000       pre-score budget record
+ 1,000,000       future-isolation record
+ 1,000,000       score-attempt record
+ 453*1,000,000   row score records
+ 16,000,000      aggregate evidence
+ 1,000,000       result
+ 1,000,000       report
+ 16,000,000      artifact manifest
```

If that conservative sum exceeds 2,000,000,000 bytes, the one-shot run terminates
before future isolation and before target scoring. The future-isolation and score
launchers set hard `RLIMIT_FSIZE` limits of 1,000,000 bytes. The supervisor
pre-serializes and bounds its other listed outputs before exclusive write. Fast
verification reconstructs the pre-score snapshot, replays the exact projection, and
checks immutable canonical bytes and the corresponding per-file bounds; the final
manifest plus its own bytes must still remain below the global ceiling.

The complete formal body runs as a transient cgroup-v2 user-systemd service with a
unique `mm009-custody-*` unit, `Type=exec`, `RuntimeMaxSec=14400s`,
`KillMode=control-group`, `KillSignal=SIGKILL`, `SendSIGKILL=yes`, and `Restart=no`.
The service starts through `env -i` with the exact frozen scientific environment and
asserts its expected cgroup before importing MM-009 or NumPy. This outer boundary
contains descendants even when an inner launcher creates a new session. A fresh
known-answer service must prove that a nested `setsid` descendant cannot survive its
root before either formal execution or semantic replay. On normal exit, timeout, or
client interruption, the supervisor kills/stops the complete unit, resets it, and
confirms it is inactive. Absence of cgroup v2, the user-systemd bus, the pinned
systemd executables, exact environment, containment proof, or inactive cleanup is a
pre-execution failure.

The frozen local resource ceilings are at most 8 concurrent predictor/replay workers,
900 seconds per predictor/replay worker, 14,400 seconds total formal wall time, and
2,000,000,000 artifact bytes. Process-specific fail-closed timeouts are 60 seconds for
an isolation probe, 300 seconds for the complete future-isolation sweep, and 120
seconds per score-only child. Exceeding a timeout, the wall ceiling, or the artifact
ceiling terminates the one-shot output; ceilings may not be relaxed after marker.

## 9. Result ladder and successor direction

Decision precedence is:

1. any parent/alignment/synthetic/future-isolation/package defect or incomplete family
   evidence -> `invalid_MM009`;
2. any real null at 6/8 in any family ->
   `invalid_MM009_real_negative_control`;
3. any real null at 3--5/8, range preemption, activity ambiguity, or mixed family
   gate ->
   `MM009_inconclusive`;
4. only with every family clear of steps 1--3, one or more complete family passes ->
   causal family support classification and GO;
5. no pass and every family meets one of the two clean bounded NO-GO diagnoses in
   section 6 -> clean bounded NO-GO with per-family mechanism diagnosis;
6. otherwise -> `MM009_inconclusive`.

Thus a passing family cannot override an invalid or inconclusive sibling family. The
stored passing-family list and `go` flag are empty in every preempted state.

On GO, the proposed direction is a **new** TAESD/MM-001 successor that encodes each
passing frozen pixel forecast through the tiny TAESD bridge and tests whether the
latent predictive objective gains the same direction under unchanged
persistence/null controls. It may not rewrite or rerun MM-001 under its old identity.

On clean NO-GO, the proposed direction is MM-010, a source-only analog/coverage
diagnosis. Its candidate question is whether leave-one-video-out historical
`(previous,current)` analogs can transfer successor/change beyond persistence under a
fixed deranged-history control, with within-video retrieval retained only as a
coverage diagnostic.

Neither direction is executable or frozen by this protocol. Before either successor
may run, it requires a new experiment identity, task, complete protocol, exact
implementation/config binding, synthetic evidence, independent pre-real audit GO,
and empty canonical output. Its own protocol must define the gates and admissible
diagnoses; the sketches above do not license a claim.

On invalid or inconclusive MM-009, run neither branch. Diagnose and preregister a new
assay first. AIDE^2 remains outside this one-shot assay. A bounded AIDE-style harness
comparison may be preregistered only after MM-009 closes and supplies a repeatable
development evaluator with a physically withheld one-shot judge; it cannot adapt on
MM-009's real future targets.

## 10. Artifacts and verification

Canonical output is
`bench/multimodal_causal_diagnostics/results/MM-009/`. Prepared records include the
protocol/config/schema, exact source/runtime and copied NumPy/stdlib/custody manifests, opaque parent copies,
supersession record, pre-marker isolation evidence, independent pre-real audit, and
freeze record. Formal records add the `0444` start and synthetic-control markers,
detached source and target inventories, post-detachment isolation evidence, 453
canonical prediction triples (`predictions.npy`, `worker-evidence.json`, and chained
`commit.json`), prediction freeze, pre-score budget projection, fitting-free
future-isolation evidence, 453
immutable score records, aggregate primitive evidence, result, report, and
last-written artifact manifest.

The fast verifier checks exact tree census, modes, hashes, no links/special files,
parent pins, marker ordering, prediction byte counts, 453-row commit chain, primitive
SSE/count arithmetic, pre-score budget replay, every declared per-file bound,
decision replay, and artifact-manifest closure without fitting.
The semantic verifier runs as a separate session under its own identical 14,400-second
total-wall ceiling, reruns and bit-compares the formal synthetic panel, reconstructs
normalizers/rows from the copied parent, reruns each source-only predictor in canonical
order, bit-compares fit and prediction records, recomputes
scores/aggregates/decision/report, and repeats the future-mutation isolation suite.

Before execution an independent results-audit pass must record GO on protocol clarity,
branch completeness, controls, exact implementation binding, synthetic evidence, and
target custody. After execution the same audit workflow independently recomputes the
quantitative/semantic claim before any branch or documentation promotion. Failed or
missing audit evidence blocks interpretation.

## 11. Abandonment criteria

Abandon this exact run before real scoring if the MM-007 package or v2.2 dependency
hashes drift; exact row/normalizer parity fails; the Landlock/seccomp sandbox cannot
deny all target, panel, repository, cross-process, and six network variants; any
synthetic or mutation gate fails; the frozen protocol/config/source/dependency set
changes after audit; or any exact ceiling in section 8 is exceeded or projected to be
exceeded. Once the formal marker exists, any interruption or defect is terminal for
that output root. A retry requires a new experiment identity, new pre-real audit, and
an empty output root.

# Architecture Decision Records

Short, dated, and honest. Each records what was decided, what was rejected, and
what would change the decision.

---

## ADR-001 - Build a vertical slice, not the eleven-service platform

**Status:** accepted

**Context.** The original concept spanned a clinical copilot, document intelligence,
lab intelligence, drug safety, guideline RAG, risk prediction, imaging metadata,
voice transcription, Kafka, Spark, Kubernetes, MLflow and Grafana. Built by one
person, that yields eleven components at roughly twenty percent depth each.

**Decision.** Ship one clinically coherent path end to end: FHIR ingest -> patient
timeline -> deterministic lab intelligence -> guideline RAG with enforced citations
-> verified clinician summary, with tracing and an evaluation harness around it.

**Rejected.** Breadth-first delivery. Feature count is not evidence of engineering
judgement, and an unjustified event bus invites the question "what load required
this?" with no good answer.

**Revisit when.** A second writer, a real EHR feed, or multi-tenant deployment
appears. Those create the coupling and throughput problems the deferred
components actually solve.

---

## ADR-002 - SQLite for v1, Postgres behind the same interface

**Status:** accepted

**Context.** One writer, roughly a thousand rows, and a hard requirement that the
repo runs with `python3` and no install step.

**Decision.** SQLite via `nullius/store.py`. All access goes through `Store`, which
exposes parameterised SQL and typed read models rather than raw connections.

**Consequences.** Swapping to Postgres means changing the connection and the
autoincrement declaration; no caller changes. What SQLite gives up - concurrent
writers, partitioning, real analytics - is not needed at this size.

---

## ADR-003 - Groundedness by token overlap, and why that is acceptable

**Status:** accepted, with stated limitations; superseded in part by ADR-008

**Context.** Every answer must be verified before a clinician sees it. The rigorous
approach is natural language inference or an LLM judge; both need a model and,
here, network access.

**Decision.** Score each sentence by the fraction of its content tokens present in
the evidence it cites. Refuse below threshold.

**Limitations, stated plainly.** Token overlap is not entailment. It will accept a
sentence that reuses the evidence's vocabulary while inverting its meaning
("potassium above 6.0 is not an emergency") or while changing a number ("withhold
above 4.9 mmol/L" against evidence saying 6.0, which scores 0.97), and it
penalises correct paraphrase. It is a *necessary* condition for groundedness, not
a sufficient one.

**Why it still earns its place.** It is deterministic, free, runs in CI, needs no
network, and catches the dominant real failure - a fluent sentence citing a
passage that does not contain it.

**Update.** The two limitations above were not left as prose caveats. Both are now
closed by explicit claim-level gates in `nullius/verification.py` (numeric support
and polarity conflict), each with its own red-team arm and its own ablation row, so
the residual weakness of this metric is bounded by something measured rather than by
a disclaimer. See ADR-008. An NLI judge remains the next upgrade for paraphrase.

---

## ADR-004 - Standard library HTTP server instead of FastAPI

**Status:** accepted

**Context.** FastAPI is the obvious choice and is what production would use, but it
is a dependency, and the repo must run in a clean sandbox with no installs.

**Decision.** `nullius/api.py` uses `http.server` with an explicit route table that
maps one-to-one onto FastAPI path operations. Authentication resolves headers into
the same `Principal` object the rest of the system uses, so RBAC and audit logging
have a single implementation regardless of transport.

**Consequences.** No automatic OpenAPI generation or request-model validation.
Migration is mechanical: each route function becomes a decorated endpoint with a
Pydantic model.

---

## ADR-005 - BM25 before embeddings

**Status:** accepted

**Context.** Vector search is assumed for RAG, but clinical queries are dense with
rare exact terms - drug names, analytes, thresholds - which lexical scoring handles
well, and the sandbox has no network for a hosted embedding model.

**Decision.** BM25 over heading-aware chunks, with the scorer injected into
`Retriever` so a vector or hybrid scorer replaces it without touching the copilot.

**Consequences.** Pure synonymy fails ("kidney failure" will not match "renal
impairment" unless both appear). Patient-context query expansion partially
compensates. The measured baseline in `eval/goldset.json` is what any embedding
upgrade must beat - which is the point of having a baseline at all.

---

## ADR-006 - No risk-prediction models on synthetic data

**Status:** accepted; refined by ADR-015, which permits a *declared-coefficient* score
(never a fitted one) on the same reasoning

**Context.** Sepsis, readmission, ICU transfer and mortality prediction were in the
original scope and are the most eye-catching features on paper.

**Decision.** Excluded entirely.

**Rationale.** Trained on synthetic records, such a model learns the generator's
rules and nothing about clinical reality. Reporting an AUROC from it would be
misleading, and any informed reviewer will say so. Deterministic, explainable lab
intelligence answers the same clinical question - "what changed and does it
matter?" - and every output can be traced to a reference range or a regression
slope.

**Revisit when.** Access to a real de-identified dataset such as MIMIC-IV, with the
appropriate data use agreement, plus a proper temporal validation split.

---

## ADR-007 - Curated interaction rules, not an LLM

**Status:** accepted

**Context.** Drug interaction checking needs a source of truth. Licensed databases
(DrugBank, First Databank) are paid; RxNorm relationships need network access.

**Decision.** A small hand-curated rule table in `nullius/interactions.py`, each rule
carrying mechanism, effect, action and source. An LLM is never asked whether two
drugs interact.

**Rationale.** A hallucinated interaction - or worse, a missed one - is precisely the
failure that makes clinical AI unsafe. A rule table is incomplete but never
invents. Incompleteness is documented; invention would be hidden.
## ADR-008 - Red-team arms and gate ablation as the way to validate a weak proxy metric

**Status:** accepted

**Context.** ADR-003 accepts a proxy metric (token overlap) with two known holes.
The original validation was a single ungrounded control provider, which is a weak
test: it fails every gate at once, so it proves the gates do *something* without
showing which gate does what, or whether a targeted attack gets through. A safety
layer only ever tested against a generator that misbehaves in the most obvious
possible way is close to untested.

**Decision.** Two changes, both cheap and both deterministic.

1. **Targeted red-team arms.** `numeric-tamper` and `polarity-tamper` wrap the honest
   extractive provider and corrupt only the payload: every sentence is verbatim
   guideline text with a valid citation marker, but a threshold is shifted or a
   recommendation is negated. Overlap-based groundedness scores these near 1.0 by
   construction, so they attack precisely the documented hole rather than a strawman.
   Tampering is realistic, not absurd - 6.0 becomes 4.9, not 600.
2. **Gate ablation.** Every gate can be disabled individually via a frozen `Gates`
   config, and the harness re-runs all arms with one gate removed at a time. Leakage,
   answers served and over-refusals are reported per configuration.

**Consequences.** The safety claim becomes falsifiable and attributable: the numeric
and polarity gates each account for 16 blocked leaks, the coverage gate is the only
one that changes what the grounded pipeline serves, and the citation and groundedness
gates show **zero incremental catch on this gold set** because the stricter gates
reach the verdict first. That last result is reported rather than hidden. It is a
real finding about ordering and redundancy, and it matters for the `openai` provider,
where free text can ignore the evidence entirely and those two gates become the only
defence.

**Cost, stated.** Two extra providers, an ablation loop that multiplies evaluation
cost by six, and a metrics correction: v1 reported the control's blocked answerable
cases as `unsafe_answers: 14`, conflating "attempts blocked" with "harm delivered".
Unsafe answers (served when they should have been blocked) and over-refusals
(refused when answerable) are now separate, because a system can only be judged on
both error types at once.

---


## ADR-009 - Synthetic dermoscopy, feature-based classification, and reporting the null result

**Status:** accepted

**Context.** The imaging path needs data. Real dermoscopy datasets (ISIC, HAM10000) cannot be
vendored into a repository that must run offline with no downloads, and no real patient
imagery may be used. So the images are generated: `nullius/lesions.py` renders lesions from a
radial Fourier boundary with vignetting, blotches and hair strokes, and writes them as real
DICOM files.

This creates an obvious hazard. A model trained and tested on data produced by code in the
same repository will score near-perfectly, and quoting that score as if it meant something is
the single most common dishonesty in portfolio machine learning.

**Decision.**

1. **A feature-based MLP, not a CNN.** Nine named morphological measurements (asymmetry,
   border irregularity, colour variegation, diameter, eccentricity and so on) feed a 9-12-1
   network. With 80 images a CNN would memorise the generator. A feature model at this scale
   is both the defensible choice and the explainable one: every input to the decision has a
   name a clinician recognises, which is what makes exact Shapley attribution meaningful
   and, at nine features, computationally trivial (ADR-012).
2. **The operating point is chosen on the training split only.** Selecting a threshold on test
   data and then reporting test sensitivity at that threshold is leakage, and it is extremely
   common. The same applies to the feature statistics behind the out-of-distribution gate.
3. **Discrimination metrics are framed as pipeline integrity, never as clinical validity.**
   The AUROC originally reported here was 1.000, which ADR-013 later identified as a defect in
   the generator rather than a result. The current figure is 0.878 on the held-out split, and it
   is presented in the README, on the site and in `report.json` as evidence that the plumbing is
   correct and as nothing else.
4. **The null result is published.** Under the original generator, `edge_gradient` separated the
   classes at a Cohen's *d* of 0.18 while the other eight ranged from 2.50 to 9.70 — and that
   9.70 was itself the smoking gun for ADR-013. Recomputed on the honest generator
   (`make separation`), the nine features span *d* 0.42 to 1.85, with `variegation` now the
   weakest at 0.42 and no single feature exceeding AUROC 0.894 on its own. The weak features are
   kept, because border sharpness and colour variegation are genuinely discriminative in real
   dermoscopy and the generator is the limitation rather than the feature, and they are reported,
   because removing a feature after seeing that it did not help is how an honest evaluation
   becomes a dishonest one.

**Consequences.** The imaging numbers are weaker evidence than they superficially appear, and
the project says so in every place they are displayed. What the imaging evaluation *does*
establish is behavioural: unreadable files, uniform frames and out-of-distribution features
are refused with specific reasons, and disarming a gate measurably changes what gets through.
Those claims survive the synthetic data, because they are claims about the pipeline rather
than about medicine.

---

## ADR-010 - Hand-written ONNX, with ONNX Runtime as the preferred backend

**Status:** accepted

**Context.** "ONNX Runtime inference" is a standard MedTech requirement, and the obvious
implementation is `pip install onnx onnxruntime` plus `torch.onnx.export`. That conflicts with
ADR-002: the repository must run and be tested with `python3` alone. It also outsources the
interesting part - what an `.onnx` file actually *is* - to a library.

**Decision.** Write the format directly, and support both backends.

- `nullius/onnx.py` contains a protobuf writer that emits a valid ModelProto by hand: varints,
  length-delimited fields, IR version 8, opset 13, initialisers as raw little-endian float32,
  a graph of MatMul / Add / Relu / MatMul / Add / Sigmoid.
- The same module contains a generic protobuf *decoder* and a pure-Python interpreter, so a
  model can be executed with zero dependencies installed.
- `load_session()` prefers `onnxruntime` when it is importable and falls back to the
  interpreter when it is not. The `VisionPipeline` records which backend served each request.

**Validation.** The decisive test is not that the parser can read what the writer produced -
that only proves self-consistency. It is that an independent production runtime accepts the
file. ONNX Runtime loads it, reports the input signature as `features [1, 9]` and the output
as `probability`, and its results agree with the pure-Python interpreter to a maximum absolute
difference of **8.870e-08 over 200 random inputs**. That test runs in CI whenever
`onnxruntime` is present and is skipped, not silently passed, when it is absent.

**Consequences.** Cross-framework export is genuinely demonstrated rather than delegated, and
the deployment story is real: the same artefact runs under the production runtime in an
environment that has it, and under the interpreter in one that does not. The cost is that only
six operators are supported, and the loader raises on anything else instead of guessing -
refusing to execute a graph it does not fully understand is the same principle the rest of the
system is built on.

## ADR-011: Judge the question, not the patient — three defects found by cold-start probing

**Status.** Accepted.

**Context.** The suite was green at 86 tests and every published metric was healthy. Probing a
freshly unpacked copy with questions the goldset never asks found three defects, each of which
produced confident, fully cited, high-groundedness output. That is the precise failure mode this
project exists to prevent, and the evaluation harness could not see any of them, because a
goldset built from answerable clinical questions never asks an unanswerable one.

1. **The coverage gate could be bypassed.** `question_coverage` returned
   `max(question_terms, clinical_signal_terms)`. The second term was added to rescue deictic
   questions ("why is this happening?"), which carry no content words of their own. But it is
   computed from the patient's record, so for any patient with an active problem list it alone
   cleared the threshold — for every question. Asked "What is the capital of France?", the
   system returned eight sentences of nephrology guidance with `refused: false` and a
   groundedness of 0.925.

2. **The extractive provider degenerated into a fixed paragraph.** Sentences were ranked by
   question overlap, but zero-overlap sentences were kept. When nothing matched, the sort key
   fell through to sentence length, so the provider emitted the two shortest sentences of every
   block. Five unrelated questions returned one identical answer, and even on-topic questions
   opened with whichever sentence was shortest rather than most relevant.

3. **The polarity gate silently disarmed itself.** It compared a claim's negation cues against
   the entire cited chunk. Chunks are hundreds of words and almost always contain a negation
   somewhere, so the set difference was usually empty. An inverted claim — "requires *not*
   prompt review" — passed unflagged whenever its own chunk mentioned "not" elsewhere.

**Decision.** Coverage now classifies the question before scoring it. A question whose topical
terms (content terms minus a small deictic list) appear nowhere in the retrieved passages or the
patient's record is out of scope and is scored on its own terms only — no context rescue.
Deictic questions have no topical terms by construction, so they keep the fallback. Generation
selects only sentences that overlap the question, with an evidence-summary pass that runs *only*
when the first selects nothing. Polarity is compared sentence-to-sentence against the closest
source sentence rather than to the pooled chunk.

**Alternatives considered.** Raising `min_question_coverage` would have refused the deictic
questions the fallback was written for, trading a false-negative class for a false-positive one.
Scoring generation against the *expanded* query was tried and rejected: it fixed the deictic case
but the added context terms outnumber the question's own, so metformin-dosing and hyperkalaemia
questions both returned whatever matched the problem list. Retrieval wants expansion; sentence
selection does not.

**Consequences.** Off-topic questions are refused as `insufficient_query_coverage`, distinct
questions receive distinct answers, and mean groundedness rose from 0.933 to 0.945 because the
selected sentences are now the relevant ones. Fixing the provider also exposed a fourth issue:
the numeric red-team arm was a no-op whenever its selected evidence contained no digits, so it
emitted legitimate text that the harness scored as gate leakage. It now injects an unsupported
figure, matching the guarantee the polarity arm already had. `tests/test_scope.py` locks all of
this down with 11 tests, including the counterweight test that deictic questions still answer.

**What this says about the evaluation.** The headline safety numbers were true and also
incomplete. A goldset of answerable questions measures how well the gates handle questions the
system is meant to answer, and says nothing about questions it is not. Both are now tested, and
the distinction is worth more than the metrics.

---

## ADR-012 - Exact Shapley attribution, because nine features is small

**Status:** accepted; replaces the leave-one-out attribution described in ADR-009

**Context.** Attribution was originally leave-one-out: replace each feature with its
training mean, re-run the model, report the drop. It is cheap and easy to explain, and
the README was careful to say it was not SHAP. It is also wrong in a specific way that
matters here: when features are correlated - and `diameter_norm`, `area_fraction` and
`eccentricity` obviously are - leave-one-out double-counts shared signal, so the parts
do not sum to the whole and no amount of caveat makes them sum.

**Decision.** Compute exact Shapley values by enumerating every coalition. Nine imaging
features is 2^9 = 512 model evaluations per served study; seven risk factors is 2^7 = 128
per patient. Both are memoised within a call.

**Rationale.** The usual objection to Shapley is cost, and the usual answer is a sampled
approximation such as KernelSHAP, which introduces its own estimation error. At this
feature count the exact computation costs milliseconds, so the approximation would be
pure downside. The efficiency axiom then gives a free correctness check that a sampled
method cannot: the attributions must sum exactly to `f(patient) - f(baseline)`. The
measured residual is ~1.1e-16, and `tests/test_explainability.py` asserts efficiency,
dummy, and symmetry directly rather than trusting the implementation.

**Consequences.** This does not scale, and the decision is explicitly bounded by that: a
pixel-level CNN would need KernelSHAP or integrated gradients. The baseline is also a
modelling choice, not a neutral fact - for imaging it is the training mean, for risk it is
every factor at its clinical reference value - and it is stated wherever attributions are
displayed, because a Shapley value is only meaningful relative to what it is compared against.

---

## ADR-013 - The AUROC of 1.000 was a bug report, not a result

**Status:** accepted

**Context.** Every version of this project up to v3 reported an imaging AUROC of 1.000 with
an ECE of 0.0005, alongside a paragraph explaining that the number was an artefact of
synthetic data. The caveat was honest. It was also insufficient, because a perfect score
was being published on the site as a headline metric with the explanation underneath it.

**Investigation.** The old generator drew benign and suspicious lesions from *non-overlapping*
parameter ranges. The classes were therefore linearly separable by construction, and
`edge_gradient`'s Cohen's *d* of 9.70 was the tell: no real morphological feature separates
melanoma from a benign naevus that cleanly. The model was not learning a hard problem well.
It was learning a trivial problem perfectly.

**Decision.** Fix the generator, not the caveat.

1. Severity is sampled from overlapping Gaussians (benign mean 0.30, suspicious mean 0.70,
   shared SD 0.15), so the classes genuinely overlap in feature space.
2. 6% label noise, because real pathology labels disagree.
3. Six of eighty cases are deliberately atypical - suspicious lesions that present benignly
   and vice versa - and are flagged in `truth_note`.

**Result.** AUROC fell from 1.000 to 0.878, sensitivity to 0.842 and specificity to 0.762,
and ECE rose from 0.0005 to 0.108. These are worse numbers and a better project. A test now
fails the build if any single feature reaches AUROC 0.99 alone, so the generator cannot drift
back into separability without someone noticing.

**Consequences.** Anyone comparing this repo to its earlier published metrics will see them
get worse. That trade is correct: a perfect score on a synthetic benchmark is evidence of a
broken benchmark, and the willingness to publish the regression is worth more than the 1.000.

---

## ADR-014 - Sensitivity-first operating point, and a calibration that barely helped

**Status:** accepted

**Context.** The operating point was previously chosen as "the threshold achieving 95%
specificity on the training split". On the honest generator this collapsed: the chosen
threshold was 1.0000, giving sensitivity 0.000. A specificity-first rule on a hard problem
will happily select a threshold that never fires.

**Decision.**

1. **Choose the threshold by sensitivity, not specificity.** The rule is now the lowest
   threshold achieving 90% sensitivity on the training split, yielding 0.4687 (specificity
   0.737 at that point). For a triage aid that flags lesions for human review, a false
   positive costs a second look and a false negative costs a missed melanoma. The asymmetry
   should be in the threshold rule, not in a footnote.
2. **Platt scaling, fitted on the training split only.** Parameters a=0.6004, b=0.6551.

**The honest result.** Calibration moved ECE from 0.108 to 0.102. That is close to nothing.
With 40 training studies there is not enough data to fit a reliable sigmoid, and reporting
"calibrated" as though it were a meaningful improvement would overstate it. It is kept
because the plumbing is correct and would matter at realistic data scale, and the before and
after figures are both published so the reader can see it barely helped.

**Consequences.** `calibrated` is a declared field, not a claim of reliability. The risk
model goes further and leaves calibration off entirely (ADR-015).

---

## ADR-015 - A declared-coefficient risk score, and why it is not fitted

**Status:** accepted; refines ADR-006

**Context.** ADR-006 cut risk prediction entirely, on the grounds that a model fitted to
synthetic data learns the generator. That reasoning still holds. But it left the project
with no answer to the question a clinical decision support system exists to answer -
"which of my patients is deteriorating?" - and the honest response to a hard question is
not always silence.

**The decisive fact.** This cohort has no outcomes. Not scarce outcomes; none. No patient
in it has progressed or not progressed, because they do not exist. Any model fitted here
would have invented both its features and its labels, and its AUROC would be a number
about a random number generator wearing the costume of clinical evidence.

**Decision.** Ship a risk score, but make it a *declared prior* rather than a fitted model.

1. Seven factors, each an additive term in log-odds, with coefficients written down in
   `nullius/risk.py` in the direction and rough magnitude KDIGO 2024 and the CKD literature
   describe. Each factor carries its `source` string into the API response.
2. Coefficients are expressed per clinically meaningful step - 0.55 per 10 mL/min of eGFR
   below 60 - so a nephrologist can disagree with a specific number rather than with a
   weight vector.
3. `validated: false` on every response, and no AUROC is computed or published for it.
4. Four abstention gates: completeness, staleness, plausibility, and an indeterminate band.

**The bug this design caught.** The first implementation refused any value outside the
coefficient's modelled range as "implausible". On the cohort this meant the sickest patients -
eGFR in the teens, potassium near 6 - were precisely the ones the tool declined to score. A
triage aid that goes quiet as the patient deteriorates is worse than no triage aid, because
silence reads as reassurance. Plausibility is now split in two: *physiologically impossible*
values (potassium of 42, a unit error) refuse, while *extreme but real* values are clamped,
flagged in a `clamped` list, and still scored. `tests/test_risk.py` locks this in.

**Abstention as a first-class output.** Missing optional factors do not force a refusal -
UACR is absent for most real CKD patients, and refusing everyone with a monitoring gap would
mean refusing exactly the patients who need review. Instead each missing factor *widens* the
indeterminate band by 0.06, so less information produces more abstention rather than
unchanged confidence. On the 12-patient cohort the score serves 7 and abstains on 5.

**Consequences.** The gate ablation is reported honestly, including the uncomfortable part:
disarming the plausibility or confidence gate changes nothing on this cohort, because no
synthetic patient triggers them. Those gates are load-bearing only in `tests/test_risk.py`,
where the inputs are constructed to hit them. Saying so is better than implying four gates
are each earning their keep in production. Replacing `FACTORS` with a fit on real outcomes
and setting `validated: true` is the only change a deployable version needs.

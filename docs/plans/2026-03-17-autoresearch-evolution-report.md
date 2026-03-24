# Autoresearch Skill Evolution Report

Date: 2026-03-17

## Executive Summary

Your current `autoresearch` skill is a disciplined single-loop experiment runner, not an actual search system. That is the core truth.

It already has:
- hard constraints;
- a mechanical keep/discard policy;
- a fixed evaluation protocol;
- lightweight run artifact templates.

It does **not** yet have:
- a candidate-generation engine;
- a reusable search memory beyond flat logs;
- failure-aware repair logic;
- portfolio/island search;
- checkpoint/resume;
- sample-efficient proposal logic for expensive runs.

My recommendation:

- **Borrow architecture from `algorithmicsuperintelligence/openevolve` first.**
  - Strategic fit: **85%**
  - Direct code-reuse fit: **30%**
- **Borrow search-policy ideas from `tennisonliu/LLAMBO` second.**
  - Strategic fit: **75%**
  - Direct code-reuse fit: **20%**
- **Borrow prompt and repair patterns from `shyamsaktawat/OpenAlpha_Evolve` third.**
  - Strategic fit: **60%**
  - Direct code-reuse fit: **25%**

The right move is **not** to merge these repos wholesale. The right move is to turn your current skill into a layered system:

1. invariant evaluator and decision policy;
2. proposal engine;
3. experiment archive with lineage and failure memory;
4. portfolio scheduler;
5. resume/checkpoint support.

## Current Baseline: What Your Skill Actually Is

Current local baseline:
- [`.claude/skills/autoresearch/SKILL.md`](/Users/a1-6/Documents/projects/hgt-local/.claude/skills/autoresearch/SKILL.md)
- [`.claude/skills/autoresearch/references/workflow.md`](/Users/a1-6/Documents/projects/hgt-local/.claude/skills/autoresearch/references/workflow.md)
- [`.claude/skills/autoresearch/references/decision-policy.md`](/Users/a1-6/Documents/projects/hgt-local/.claude/skills/autoresearch/references/decision-policy.md)
- [`.claude/skills/autoresearch/references/constraints.md`](/Users/a1-6/Documents/projects/hgt-local/.claude/skills/autoresearch/references/constraints.md)

Strengths:
- the evaluation target is fixed;
- the optimization target is singular enough to automate;
- the allowed edit surface is constrained;
- the skill already separates workflow, constraints, and policy.

Weaknesses:
- proposal quality is mostly human-authored heuristics;
- no structured memory of failure types, lineage, or novelty;
- no multi-armed search strategy;
- no explicit crash-repair path;
- no checkpoint/resume model;
- no search abstraction separating config search from code mutation.

Important design bug:

The current workflow says discard via `git reset --hard HEAD~1`. In a collaborative or dirty worktree, that is unsafe. Probability this bites you later if autonomy increases: **80%**.

If you evolve this skill, fix that first. Discard should mean:
- move baseline pointer;
- record the failed child in the run archive;
- avoid destructive history rewriting unless the worktree is guaranteed isolated.

## Repo 1: `algorithmicsuperintelligence/openevolve`

Repo:
- [algorithmicsuperintelligence/openevolve](https://github.com/algorithmicsuperintelligence/openevolve)

Observed freshness from cloned repo:
- latest local clone commit: `65cbbe839d2cdb14185a8b594dd89f330e77d8a3`
- commit date: **2026-02-04**

Observed maturity signals:
- about **54** test files;
- about **214** Python files;
- active config system;
- checkpointing, tracing, evaluator artifacts, islands, novelty handling.

### What OpenEvolve is really good at

OpenEvolve is the strongest donor because it solves the problem your skill does not yet solve: **how to maintain a diverse, persistent search process over code candidates**.

Reusable ideas:
- island-based search instead of one linear branch;
- archive/database of programs with lineage and metrics;
- prompt context built from parent, top programs, and inspirations;
- diff-based mutations instead of full rewrites;
- artifact side-channel: failures become prompt input;
- checkpoint/resume;
- early stopping;
- novelty filtering and diversity pressure;
- parallel evaluation controller.

### What to steal

Steal these ideas almost directly at the architecture level:

1. **Program / experiment archive**
   - Every experiment should have:
   - run id, parent id, hypothesis class, files touched, config diff, metrics, failure signature, artifact summary, keep/discard decision.

2. **Artifact-aware prompting**
   - For HGT runs, the useful artifacts are:
   - OOM errors;
   - tensor shape mismatches;
   - NaN/instability traces;
   - timeout markers;
   - final metric block.

3. **Portfolio / island search**
   - Not 5-15 heavy concurrent islands like OpenEvolve.
   - For this repo, use **3 lightweight strategy islands**:
   - `config-tuning`
   - `stability-and-regularization`
   - `architecture-tweaks`

4. **Checkpoint / resume**
   - This is mandatory once runs take tens of minutes or more.

5. **Novelty gate**
   - Prevent repeated equivalent experiments.
   - For your repo this can start simple:
   - normalized config hash;
   - touched-file signature;
   - failure-signature hash.

### What not to steal directly

- full MAP-Elites implementation;
- large population sizes;
- generic code-evolution assumptions;
- LLM quality feedback mixed into fitness;
- complex process-parallel machinery.

Why:
- your evaluations are expensive;
- your search space is narrower;
- your success metric is already mechanical;
- population-heavy evolutionary search is wasteful when one evaluation can cost up to an hour.

### My call

OpenEvolve should be your **architectural spine**.

Probability it contributes materially if used as a concept donor: **85%**  
Probability a direct code transplant is worth the pain: **30%**

## Repo 2: `tennisonliu/LLAMBO`

Repo:
- [tennisonliu/LLAMBO](https://github.com/tennisonliu/LLAMBO)
- [Paper / OpenReview](https://openreview.net/forum?id=OOxotBmGol)

Observed freshness from cloned repo:
- latest local clone commit: `196fe237f60a3d3a2fa53cbf8f474ec20a01dd57`
- commit date: **2024-12-17**

Observed maturity signals:
- research-code shape, not production framework;
- many experiment scripts;
- older OpenAI client assumptions;
- HPO/tabular benchmark framing.

### What LLAMBO is really good at

LLAMBO is not a code-evolution framework. It is a **sample-efficiency idea** for black-box optimization.

Its strongest concepts:
- LLM-assisted zero-shot warm start;
- LLM-generated candidate proposals;
- LLM surrogate scoring of candidate points;
- expected-improvement style selection;
- input warping for awkward numeric ranges;
- dedup/filtering of proposed candidates.

That matters because your HGT training loop is expensive. When evaluations are expensive, brute-force evolution is dumb. Sample efficiency matters more than diversity theater.

### What to steal

1. **Warm-start candidate generation**
   - Before running experiments, generate 5-10 ranked config candidates from:
   - search space definition;
   - current best run;
   - previous failures;
   - known priors.

2. **Range-aware prompting**
   - Express the config search space explicitly in the prompt.
   - Include discrete, ordinal, float, and log-scale metadata.

3. **Candidate de-duplication**
   - Reject repeats before running.

4. **Early-stage search policy**
   - Use LLAMBO-like proposal logic in the first 10-20 trials where historical data is sparse.

5. **Input warping**
   - Especially useful for:
   - learning rate;
   - weight decay;
   - warmup ratio;
   - hidden size / head count ratios.

### What not to steal directly

- its old OpenAI API plumbing;
- tabular benchmark code;
- full surrogate stack;
- direct dependence on its experiment framework.

Why:
- the code is older and benchmark-specific;
- your task is not generic tabular HPO;
- you do not need the whole surrogate-model research stack to get 70% of the value.

### My call

LLAMBO should be used as a **proposal-policy donor**, not as a framework dependency.

Probability it improves the first phase of HGT search if simplified aggressively: **75%**  
Probability direct repo integration is worth it: **20%**

## Repo 3: `shyamsaktawat/OpenAlpha_Evolve`

Repo:
- [shyamsaktawat/OpenAlpha_Evolve](https://github.com/shyamsaktawat/OpenAlpha_Evolve)

Observed freshness from cloned repo:
- latest local clone commit: `0389aea0e0a745b61babdc0fdd18ac61098dfa84`
- commit date: **2025-05-31**

Observed maturity signals:
- about **1** test file;
- about **19** Python files;
- modular agent layout;
- diff-oriented mutation and bug-fix prompts;
- Docker sandbox evaluation.

### What OpenAlpha_Evolve is really good at

OpenAlpha_Evolve is weaker than OpenEvolve on systems rigor, but it is cleanly modular and very explicit about agent roles.

Its best ideas for your case:
- separate prompt types for initial generation, mutation, and bug fixing;
- diff-based edits as a first-class mutation format;
- explicit task manager / evaluator / selector / database boundaries;
- evaluation feedback shaped for repair prompts.

### What to steal

1. **Crash-repair mode**
   - If a run fails with a known class of error, do not route it through the same “improve metric” prompt.
   - Route it through a targeted repair prompt.

2. **Prompt taxonomy**
   - Split prompts into:
   - `proposal`
   - `mutation`
   - `bug_fix`
   - `plateau_breaker`

3. **Diff discipline**
   - Require localized diffs for code mutations.
   - This reduces random rewrites and keeps lineage legible.

4. **Agent boundary design**
   - Even if you keep this as one skill, think in components:
   - proposer;
   - evaluator;
   - archive;
   - selector.

### What not to steal directly

- Docker code sandboxing;
- its in-memory/JSON database;
- its small-population default logic;
- most of its implementation internals.

Why:
- your repo evaluates model training, not arbitrary generated algorithm code;
- Docker sandboxing is not the bottleneck;
- persistence and selection logic are less mature than OpenEvolve.

### My call

OpenAlpha_Evolve is a **prompt-and-repair donor**.

Probability it helps if limited to crash-repair and diff prompts: **60%**  
Probability it should define the main architecture: **25%**

## Synthesis: What the Next Autoresearch Should Become

### Target Architecture

Build `autoresearch-v2` as five layers.

#### 1. Invariants layer

Keep from current skill:
- hard constraints;
- validation-only decisions;
- fixed metric extraction;
- repository-specific boundaries.

#### 2. Proposal layer

Three proposal modes:

- **Mode A: config proposer**
  - LLAMBO-lite
  - proposes candidate `configs/config.yaml` changes only

- **Mode B: repair proposer**
  - OpenAlpha-style bug-fix mode
  - only activates after crash classes like OOM, NaN, shape mismatch, timeout

- **Mode C: code mutation proposer**
  - OpenEvolve-style diff mutation
  - only after config search plateaus

#### 3. Archive layer

Replace flat `results.tsv` as the only memory with a richer experiment ledger:

```text
outputs/research/<run-tag>/
├── progress.md
├── results.tsv
├── session.log
├── archive.jsonl
├── best_snapshot.json
└── checkpoints/
```

Each archive record should include:
- experiment id;
- parent id;
- island / strategy;
- hypothesis type;
- config diff;
- code diff summary;
- metrics;
- crash class;
- artifact summary;
- keep/discard;
- commit hash or patch id.

#### 4. Scheduler layer

Use **strategy islands**, not process-heavy compute islands:

- island 1: `config-exploration`
- island 2: `stability-and-regularization`
- island 3: `architecture-edits`

Migration means:
- best idea summary from one island becomes inspiration context for another;
- not literal concurrent cross-process migration at first.

#### 5. Selector layer

Decision policy should become:

1. reject duplicates / near-duplicates;
2. reject invalid experiments;
3. if crash: archive + classify + decide whether repair path triggers;
4. if metric improves: keep;
5. if plateau persists: switch island or switch proposer mode;
6. if no progress for N rounds: early stop.

## What To Build First

### Phase 0: Fix the foundation

Do this before importing any outside idea.

Deliverables:
- remove destructive discard behavior from the workflow;
- define experiment record schema;
- define crash taxonomy;
- add checkpoint/resume layout to skill docs.

Why:
- without this, more autonomy just means faster chaos.

Probability Phase 0 is necessary before anything else: **95%**

### Phase 1: Add LLAMBO-lite candidate generation

Scope:
- config-only search;
- no code mutation;
- no LLM evaluator;
- 5-10 proposed candidates per round;
- run top 1-3 after dedup/filtering.

Why first:
- lowest integration risk;
- directly targets your expensive black-box regime;
- gives immediate sample-efficiency gains.

Success criterion:
- find a better validation result within 10-20 experiments more often than the current heuristic-only loop.

Probability this is the best first experiment: **80%**

### Phase 2: Add crash-repair mode from OpenAlpha ideas

Scope:
- activate only on crash;
- prompt contains:
  - touched files,
  - error signature,
  - last successful baseline,
  - minimal diff requirement.

Why second:
- it converts dead experiments into structured recovery attempts;
- it is safer than free-form code evolution.

Success criterion:
- reduce repeated crash loops by at least 30%.

### Phase 3: Add OpenEvolve-lite archive and strategy islands

Scope:
- archive JSONL;
- 3 strategy islands;
- novelty filter;
- plateau-aware switching;
- checkpoint/resume.

Why third:
- this is where the system becomes a real search engine instead of a script.

Success criterion:
- better coverage of search space with fewer duplicate experiments.

### Phase 4: Add controlled code evolution

Scope:
- only after config search plateaus;
- only diff-based edits;
- only inside currently allowed files;
- only one mutation class at a time.

Why last:
- code mutation is high-variance and expensive;
- if done too early it will dominate cost and bury signal.

## What I Would Explicitly Not Do Yet

Do **not** do these in v1 of the upgrade:

- full MAP-Elites grid search;
- large population evolutionary loops;
- multi-model LLM ensembles;
- arbitrary-code sandbox execution;
- LLM-based fitness scoring;
- direct import of the external repos as dependencies.

These are seductive, but for this repo they are more likely to add complexity than performance.

Probability they slow you down more than they help in the next iteration: **70-85%**

## Concrete Merge Plan Into The Current Skill

### Files to change first

1. [`.claude/skills/autoresearch/SKILL.md`](/Users/a1-6/Documents/projects/hgt-local/.claude/skills/autoresearch/SKILL.md)
   - add explicit modes:
   - `config_search`
   - `repair`
   - `code_mutation`
   - `portfolio`

2. [`.claude/skills/autoresearch/references/workflow.md`](/Users/a1-6/Documents/projects/hgt-local/.claude/skills/autoresearch/references/workflow.md)
   - replace destructive discard;
   - add checkpoint/resume;
   - add island/strategy switching rules;
   - add crash-repair branch.

3. [`.claude/skills/autoresearch/references/decision-policy.md`](/Users/a1-6/Documents/projects/hgt-local/.claude/skills/autoresearch/references/decision-policy.md)
   - add duplicate rejection;
   - add plateau thresholds;
   - add crash-to-repair routing;
   - add early stop rules.

4. New reference file:
   - `references/search-architecture-v2.md`

5. New scripts:
   - `scripts/suggest_config_candidates.py`
   - `scripts/archive_experiment.py`
   - `scripts/classify_failure.py`
   - `scripts/resume_session.py`

### Minimal v2 interface

The skill should support commands like:

- “run config-search round”
- “repair last failed experiment”
- “switch to architecture island”
- “resume research from checkpoint”
- “show search archive summary”

## Recommended Order of Adoption

If you want the shortest path with the highest expected return:

1. **OpenEvolve architecture**
2. **LLAMBO proposal policy**
3. **OpenAlpha repair prompts**

If you want the lowest engineering risk:

1. **LLAMBO-lite**
2. **OpenAlpha repair mode**
3. **OpenEvolve-lite archive/islands**

My actual recommendation for this repo:

1. fix current skill foundation;
2. add LLAMBO-lite config proposer;
3. add OpenEvolve-lite archive/checkpoints/portfolio;
4. add OpenAlpha-style repair prompts;
5. add controlled code mutation last.

## Final Call

The winning hybrid is:

- **OpenEvolve for the skeleton**
- **LLAMBO for sample-efficient proposal**
- **OpenAlpha_Evolve for repair prompts and diff discipline**

Not the other way around.

If you try to make your current skill “more autonomous” before adding archive, failure memory, and proposal logic, you will mostly get a faster random walk.

That is the blunt truth.

## Sources

Primary sources used:

- Current local skill:
  - [`.claude/skills/autoresearch/SKILL.md`](/Users/a1-6/Documents/projects/hgt-local/.claude/skills/autoresearch/SKILL.md)
  - [`.claude/skills/autoresearch/references/workflow.md`](/Users/a1-6/Documents/projects/hgt-local/.claude/skills/autoresearch/references/workflow.md)
  - [`.claude/skills/autoresearch/references/decision-policy.md`](/Users/a1-6/Documents/projects/hgt-local/.claude/skills/autoresearch/references/decision-policy.md)
  - [`.claude/skills/autoresearch/references/constraints.md`](/Users/a1-6/Documents/projects/hgt-local/.claude/skills/autoresearch/references/constraints.md)

- External repos:
  - [algorithmicsuperintelligence/openevolve](https://github.com/algorithmicsuperintelligence/openevolve)
  - [tennisonliu/LLAMBO](https://github.com/tennisonliu/LLAMBO)
  - [shyamsaktawat/OpenAlpha_Evolve](https://github.com/shyamsaktawat/OpenAlpha_Evolve)
  - [LLAMBO paper](https://openreview.net/forum?id=OOxotBmGol)

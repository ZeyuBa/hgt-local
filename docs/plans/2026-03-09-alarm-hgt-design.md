# Alarm HGT Synthetic Data And Training Design

## Scope

Implement the functionality described in [model_design.md](/Users/a1-6/Documents/projects/hgt-local/model_design.md) on top of the existing `pyHGT` codebase, with one explicit extension: because no real transformed dataset exists in the workspace, the project will generate `transformed_train.json`, `transformed_val.json`, and `transformed_test.json` from a graph-based synthetic data pipeline.

The validated design below treats sections 1-5 of `model_design.md` as the effective product spec. The truncated `## 6. 技术卡点` section is ignored.

## Data Generation Strategy

Each sample is a complete heterogeneous graph representing one fault or risk event. Samples are generated from a site-level backbone plus site-local stars:

- Each `phy_site` owns exactly one `router`.
- Each site owns one or more `wl_station` nodes.
- Intra-site connectivity uses `co_site_ne_ne`.
- Inter-site router connectivity uses `cross_site_ne_ne`.

Topology generation is rule-driven rather than purely random:

- Build a connected backbone tree over sites.
- Add a small number of cross-branch backup links.
- Keep the average site degree in a narrow operating range so propagation remains interpretable.

Each sample chooses 1-3 `fault_or_risk` sites. Each chosen site independently receives one fault mode:

- `mains_failure`
- `link_down`

Noise injection is applied only to non-anchor, non-AN, still-connected sites by setting `mains_failure=1` without creating downstream impact.

## Full-Graph Invariant

Exported training and test samples must remain complete graphs.

Fault simulation is internal to the labeling engine only. It uses logical masks:

- `inactive_ne_ids`
- `blocked_edge_ids`

These masks affect reachability and label computation, but no real node or edge is removed from the exported sample. This preserves the full-graph training assumption and prevents accidental topology truncation.

## Sample Format

The transformed datasets are stored as JSON Lines files, one graph sample per line. The canonical sample schema contains:

- `sample_id`
- `seed`
- `fault_or_risk_sites`
- `nodes`
- `edges`
- `alarm_entities`

`nodes` stores only atomic attributes, not the final 32-dim feature vector. Feature construction happens at load time so feature logic remains single-sourced.

`edges` contains only forward business edges:

- `ne_alarm_entity`
- `alarm_entity_alarm`
- `co_site_ne_ne`
- `cross_site_ne_ne`

Reverse edges and self-loops are synthesized by the dataset loader.

## Labeling Rules

Label generation is driven by graph reachability from AN roots plus local deterministic alarm rules.

For `mains_failure`:

- The fault site gets `mains_failure=1`.
- Its router gets `device_powered_off=1`.
- All `wl_station` nodes in the fault site get `ne_is_disconnected=1`.
- During logical propagation, the site router is treated as inactive.
- Any downstream `wl_station` that loses AN reachability also gets `ne_is_disconnected=1`.

For `link_down`:

- The fault site and its primary upstream neighbor receive `link_down=1`.
- During logical propagation, the selected inter-site link is treated as blocked.
- Any `wl_station` that loses AN reachability gets `ne_is_disconnected=1`.

Noise sites only receive `mains_failure=1` and never trigger downstream disconnection.

## Feature Construction

All node types share a 32-dimensional feature vector, constructed at dataset load time from the complete graph topology.

Key rules:

- Distance features are computed on the full NE topology, not the logically failed topology.
- `AlarmEntity` inherits all site and NE structure features from its owning NE.
- `AlarmEntity` overwrites role bits, fills `alarm_id` one-hot, and fills normalized `domain`.
- `Alarm` nodes only carry template-level features.

Distance buckets and degree features follow the definitions in [model_design.md](/Users/a1-6/Documents/projects/hgt-local/model_design.md).

## Mini-Batch And Padding

Each stored sample remains a real complete graph. Padding exists only inside the collator.

Batching works by:

1. Grouping similarly sized graphs with a bucket sampler.
2. Padding only the `AlarmEntity` dimension to the batch-local `max_ae`.
3. Materializing padding `AlarmEntity` nodes as isolated nodes with zero features and self-loop only, while reusing the first real `NE` in the sample as their `owner_ne_index` for tensor pairing only.

Important constraint: padding `AlarmEntity` nodes must not introduce any `NE -> AlarmEntity` or `AlarmEntity -> NE` edges. The reused owner index is bookkeeping for the bilinear scorer, not a graph edge.

Padding is temporary and never persisted into transformed dataset files.

## Training Mask

The original `trainable_mask` rule from the design document is tightened.

Only `ne_is_disconnected` alarm entities are trainable, and only when their owning site is none of:

- fault or risk anchor
- AN site
- padding site

Final rule:

`trainable_mask = is_ne_is_disconnected & ~owner_is_fault_or_risk_anchor & ~owner_is_an & ~owner_is_padding`

Important constraint: `owner_is_fault_or_risk_anchor`, `owner_is_an`, and `owner_is_padding` are site-level properties. They are broadcast to every NE and AlarmEntity inside the site.

## Model And Training

The encoder remains based on `pyHGT`:

- shared 32-d input features
- type-specific input projections
- 4 HGT layers
- 64 hidden size
- 4 heads

The output head is replaced with a bilinear `EdgePredictor` over paired NE and AlarmEntity embeddings:

`score = normalize(h_ne)^T W normalize(h_ae)`

Training is implemented with a HuggingFace-style `PreTrainedModel` wrapper plus custom data collation, sampling, loss masking, and metrics.

## Metrics

Metrics are computed only on the masked trainable positions.

Edge-level metrics:

- AUC
- AP
- F1 at 0.5
- best F1 by threshold scan
- P@K
- Recall@K
- nDCG@K
- MRR

Graph-level metrics:

- Graph Accuracy
- Graph Perfect/1FP

## Testing Strategy

Tests must lock down the following invariants:

- exported samples always preserve the complete graph
- propagation uses logical masks rather than destructive graph edits
- padding only appears in the collator
- site-level attributes are broadcast consistently to all NE and AlarmEntity nodes
- trainable mask follows the final site-level rule
- synthetic labels match reachability-based expectations
- model forward pass returns correctly shaped logits and loss on padded batches

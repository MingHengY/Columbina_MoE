# Columbina 4.0-MoE Architecture Contract

## Preserved trust boundary

- C1/C2/C3 use the unchanged strict ten-fold 8:1:1 split protocol from v3.
- Gene mappings, omics preprocessing, and cancer vocabularies are fitted on training data only.
- Validation and test retain independent `train+validation` and `train+test` contexts.
- The static KG may describe held-out genes but never contains SL validation/test labels.

## Experts

### SL/omics expert

- Encodes the 262-dimensional omics vector with a LayerNorm MLP.
- Aggregates only positive SL edges from the training split with GraphSAGE.
- Validation and test never use evaluation SL labels as message-passing edges.
- Held-out genes without training SL context fall back to their omics representation.
- The learned SL-context contribution is capped so every known gene retains a
  direct omics path; this reduces the train-to-C3 representation shift.

### KG expert

- Projects every heterogeneous node type to a shared hidden dimension.
- Uses HGT layers over the schema union registered before optimizer construction.
- Adds reverse message-passing relations so genes can receive evidence from every incident KG relation.
- Aligns HGT Gene embeddings to the stage-local SL node order.

## Cancer conditioning and pair prediction

- Cancer type is represented by a learned embedding.
- Missing cancer labels use `<GLOBAL>`; cancer types unseen during training use `<UNK>`.
- Static DepMap cell-line metadata is used to fill cancer type when available.
- During training, known-cancer coverage is downsampled to the same rate in both
  label classes. If either class has no known cancers, cancer conditioning falls
  back to `<GLOBAL>` for that epoch, preventing annotation availability leakage.
- Independent FiLM layers condition the SL/omics and KG pair representations.
- Both expert decoders use the symmetric representation
  `[a+b, |a-b|, a*b, score_sum, |score_difference|]`.
- A pair-level softmax gate combines the two expert logits using expert representations,
  cancer context, unseen-gene status, SL coverage, KG coverage, and omics availability.
- In strict-inductive training, a train-only node mask periodically creates pseudo-new
  genes. Their SL message-passing edges are removed while omics, static KG, and training
  labels remain available. A 50% node mask produces a mixture of C1-, C2-, and C3-like
  pairs without exposing validation or test data.

## Objective

The training objective is:

`L = L_final + lambda_sl L_sl + lambda_kg L_kg + lambda_rank L_rank + lambda_sym L_sym + lambda_route L_route`

- `L_final`, `L_sl`, and `L_kg` are class-weighted BCE-with-logits losses.
- `L_rank` ranks positive pairs above sampled negative pairs.
- `L_sym` checks pair-order consistency; the symmetric decoder makes it zero by construction.
- `L_kg` is reliability-weighted so pseudo-new pairs with KG coverage receive more
  KG-expert supervision.
- `L_route` uses a soft coverage-aware target. It remains neutral for ordinary
  training pairs and increasingly favors KG when novelty rises, SL coverage falls,
  and KG coverage is present.
- Fixed batch-level 50:50 gate balancing is disabled; it cannot express the different
  reliability conditions of C1, C2, and C3.

## Checkpoints

Checkpoints include architecture version `4.0-MoE`, the HGT schema, cancer vocabulary,
training gene IDs, and positive training SL context. V3 checkpoints are not compatible.

## Primary evaluation metrics

- AUC: ROC area.
- AUPR: `average_precision_score`; `auprc` remains as a compatibility key.
- F1: computed with the validation-selected threshold in strict inductive evaluation.
- Precision@10: positive fraction among the ten highest-scoring pairs, independent of threshold.
- Checkpoint selection and early stopping use validation-only
  `0.4 * AUC + 0.4 * AUPR + 0.2 * F1`. Precision@10 is reported but is not used for
  selection because it is coarse on ten examples.

## Run

From the `code_v4` directory, the existing entry points remain valid:

```bash
python main.py --scenario C2 --cv --folds 10
python main.py --scenario C3 --cv --folds 10
python infer.py --help
```

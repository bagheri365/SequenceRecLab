# Matched Transformer ablation

Milestone 6 introduces the matched pair used to estimate explicit positional value.

## Identification target

The primary comparison is

\[
\text{PositionalGain@K}
= \text{Metric(SASRec)} - \text{Metric(PositionlessSASRec)}.
\]

The pair is intentionally narrower than a broad "Transformer vs. baseline" comparison. Both variants share the same item embeddings, hidden size, attention heads, Transformer depth, feed-forward width, optimizer family, objective, training examples, candidate set, output parameterization, and masked-mean readout.

## Why there is no causal attention mask

A triangular causal mask is itself positional structure: token 1 can attend to a different set of tokens than token 3 even if no positional embedding is supplied. That would make a supposedly positionless control order-sensitive.

SequenceRecLab therefore gives **both** variants full self-attention over the already-observed history prefix. The prediction target is outside that prefix, so this does not expose the target or future events. Both variants then masked-mean-pool the contextualized history states.

This means the implementation is best described as a **SASRec-style next-item Transformer ablation**, not a byte-for-byte reproduction of the original causal SASRec training recipe. The design choice is deliberate: identification of explicit positional information takes priority over reproducing every original implementation detail.

## PositionlessSASRec

`PositionlessSASRec` has a position-embedding table with the same shape as the position-aware model, but it is fixed to zero and excluded from optimization. With no positional input, no causal mask, and masked-mean readout, the model is permutation invariant with respect to an already-selected history window.

Repeated items remain repeated tokens, so multiplicity is retained.

## SASRec

`SASRec` uses the same network and readout, but its positional embedding table is learned. Swapping items between positions can therefore change the contextualized representation and next-item scores.

## History windows

Neither model silently truncates long histories. History-length selection is part of the data/example pipeline and must happen before model scoring or training. This keeps comparisons at history lengths such as 2, 3, 5, and 10 explicit and auditable.

## Training objective

Both variants use full-catalog cross-entropy over the training item vocabulary. The output logits are tied to the item-embedding matrix, excluding padding row 0. This is compatible with the primary full-catalog evaluation contract.

Training is performed in deterministic shuffled minibatches (default batch size 256). This bounds the full-catalog logit tensor to `batch_size × item_count` instead of materializing logits for every training example at once. Both matched variants use the identical batching rule and seed, so batching does not introduce an architectural difference between them.

## Reproducibility

Model initialization and training use the configured PyTorch seed. Dropout is disabled during evaluation. Primary experiments should still run the study-level three seeds and report mean/std plus paired bootstrap uncertainty for key metric differences.

## Validation checkpoint selection

Formal Transformer runs use validation-based checkpoint selection rather than a manually chosen fixed epoch. The predeclared selection metric is full-catalog **NDCG@10**, computed with the same ascending-item-ID tie rule as the main evaluator. The default maximum budget is 50 epochs, patience is 5 epochs, and `min_delta = 0.0`. After training stops, the model restores the parameters from the epoch with the best validation NDCG@10.

The validation examples use the same fixed-max-history population rule as reported evaluation: for a declared history grid such as `[2, 3, 5]`, one history-5-eligible validation cohort is selected first and then truncated to each requested history length. This prevents checkpoint selection from silently changing populations across history conditions.

For a validation run, the validation split is both the checkpoint-selection split and the reported diagnostic split; those metrics are therefore model-selection diagnostics, not final held-out estimates. For a test run, checkpoint selection still uses validation, while `test.jsonl` is used only after the best validation checkpoint has been restored. The test split must not be used to choose epoch count, patience, or other hyperparameters.

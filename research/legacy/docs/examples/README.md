# Illustrative contract fixtures

These JSON files document valid shapes only. Probabilities are invented examples, not model outputs or measured results. The sample and audit IDs are illustrative and do not constitute a verified frozen dataset. No tokenizer IDs or artifact hashes are fabricated as evidence.

The normative shape schemas live in [../contracts](../contracts/request.schema.json); semantic constraints and byte limits live in [interfaces](../interfaces.md), [dataset design](../dataset-design.md) and [tokenizer](../tokenizer.md). Implementation must share these fixtures between C++ and Python validators and add invalid/mutation cases.

Schema `$id` uses the reserved `.invalid` domain as a local identifier, never as a network dependency. Resolve references from the local contract registry; validation must not fetch remote schemas.

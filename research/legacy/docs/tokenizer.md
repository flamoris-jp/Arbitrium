# Tokenizer and input assembly

## Selected default

Use SentencePiece BPE with an Arbitrium-trained vocabulary, loaded through its native C++ processor. Python may invoke the SentencePiece trainer as a data-preparation tool; it must not implement an alternative tokenizer or model. Save `tokenizer.model`; a JSON tokenizer from another engine is not interchangeable.

Train on **train input fields only** from the frozen split assignment. Exclude dev, both calibration subsets, sealed test, rationale, targets and audit metadata. Sort samples by ID and fields in the order below. Desired vocabulary 8192 with `hard_vocab_limit=false`; store the actual size. `character_coverage=1.0`, `byte_fallback=true`, BPE, identity normalization, `add_dummy_prefix=false`, `remove_extra_whitespaces=false`, `split_by_whitespace=true`, `shuffle_input_sentence=false`, `input_sentence_size=0`, trainer thread count 1. Record all resolved trainer options and exact version; persist the actual model bytes rather than promising regeneration is bit-identical across releases.

Special IDs: PAD=0, UNK=1, BOS=2, EOS=3. Register control symbols in the order CLS, TASK, POLICY, STATE, QUESTION; read and record their actual IDs from the trained model. Training/serving fail if any required symbol is missing or collides. BOS is reserved but unused in v1 assembly. Byte fallback preserves encodability, not multilingual ability. Non-English use is outside the model claim.

## Request preprocessing

Validate UTF-8 and reject NUL and control characters except TAB/LF/CR. Convert CRLF and CR to LF; otherwise preserve case, punctuation, spacing, numbers and Unicode. Do not lowercase or apply implicit Unicode normalization. Empty/whitespace-only state, policy or question fails. Limit total JSON line to 64 KiB, state 16 KiB, policy 8 KiB, question 2 KiB (UTF-8 byte counts). Limits are checked before tokenization; the 256-token limit is checked afterwards. Over-limit input errors, never silently truncates.

Task spec supplies the exact canonical question and canonical policy text. Request question and policy must both match those canonical strings after the line-ending normalization above. Policy paraphrases and dynamically supplied rule variants are rejected in v1; a semantic policy change requires a new TaskSpec/task ID and artifact.

## Sequence construction

Insert control IDs directly; encode each content field separately with SentencePiece sampling disabled:

`CLS TASK encode(task_id) POLICY encode(policy) STATE encode(state) QUESTION encode(question) EOS`

Segments: CLS/TASK/task_id = 0; POLICY and policy = 1; STATE and state = 2; QUESTION/question/EOS = 3. PAD uses segment 0. Right-pad to the longest sequence in the batch, at most 256. Positions start at 0 including controls. Choice IDs/order are validation/output mapping only; their semantics are fixed by task spec and do not alter tokens. The fixed task's labels must not be dynamically redefined in request text.

Literal strings such as `<STATE>` inside content must remain ordinary content; only adapter-inserted control IDs delimit fields. SentencePiece control-symbol behavior is verified with fixtures. PAD cannot appear as a real token. Token cache key is SHA-256 of tokenizer bytes + assembly version + canonical normalized input JSON; changing tokenizer or task requires a new cache.

## Verification

C++ and Python's SentencePiece binding must yield identical token IDs for the fixture corpus (apostrophes, negation, decimals, comparison operators, multiline text, Unicode punctuation, literal control symbols, unusual bytes represented as valid UTF-8). Golden IDs are generated only once the implementation has produced a real tokenizer artifact; no invented IDs are checked in as evidence. Validate padding and boundary limits at 255/256/257 assembled tokens.

Source capabilities were checked against [SentencePiece options](https://github.com/google/sentencepiece/blob/master/doc/options.md) and [special symbols](https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md). These options are project choices, not claims that the current scaffold supports them.

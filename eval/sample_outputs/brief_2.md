# Technical Brief

**Question.** What retrieval strategy best reduces hallucinated citations in RAG?

_Decision impact: medium. Route: SYNC_REVIEW._

## Findings

### What approaches exist for retrieval strategy reduces hallucinated citations?
- A model writes a hypothetical answer document whose embedding retrieves real neighbors, with the dense encoder filtering hallucinated specifics. [[1]] (grounding 0.31, confidence 0.68)

### What does empirical evidence say about retrieval strategy reduces hallucinated citations?
- A model emits reflection tokens to retrieve on demand and to grade relevance, support, and usefulness of evidence. [[2]] (grounding 0.28, confidence 0.62)

### What are the key trade-offs and failure modes for retrieval strategy reduces hallucinated citations?
- A lightweight retrieval evaluator labels results correct, ambiguous, or incorrect and triggers corrective actions, including widening the search. [[3]] (grounding 0.30, confidence 0.65)

### What reliability and failure-mode considerations apply to retrieval strategy reduces hallucinated citations?
- Interleaving reasoning traces with task actions lets a model plan, act, and update plans against external tools. [[4]] (grounding 0.29, confidence 0.64)
- Injecting a small number of malicious texts per target question into a large retrieval corpus can force attacker-chosen answers with high success, showing that retrieved content is not trustworthy merely because it was retrieved. [[5]] (grounding 0.28, confidence 0.63)

## Contradictions

- **[OPEN]** on _What does empirical evidence say about retrieval strategy reduces hallucinated citations?_: Comparable evidence -> follow-up retrieval / human adjudication required
- **[OPEN]** on _What does empirical evidence say about retrieval strategy reduces hallucinated citations?_: Comparable evidence -> follow-up retrieval / human adjudication required
- **[OPEN]** on _What reliability and failure-mode considerations apply to retrieval strategy reduces hallucinated citations?_: Comparable evidence -> follow-up retrieval / human adjudication required

## Open questions / caveats

- 4 claim(s) dropped by the grounding gate.

## References

1. Gao et al. (2023). Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE). ACL. arXiv:2212.10496.
2. Asai et al. (2024). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. ICLR. arXiv:2310.11511.
3. Yan et al. (2024). Corrective Retrieval-Augmented Generation (CRAG). arXiv. arXiv:2401.15884.
4. Yao et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. ICLR. arXiv:2210.03629.
5. Zou et al. (2025). PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation. USENIX Security. arXiv:2402.07867.

---
_Every claim above resolves to a cited source above the grounding threshold; unresolved sub-questions are labelled, not answered._
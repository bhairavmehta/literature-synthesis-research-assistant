# Technical Brief

**Question.** When does iterative self-reflection improve an agent's reasoning?

_Decision impact: low. Route: AUTO_PASS._

## Findings

### What approaches exist for does iterative self reflection improve?
- A model emits reflection tokens to retrieve on demand and to grade relevance, support, and usefulness of evidence. [[1]] (grounding 0.33, confidence 0.74)
- Agents store verbal self-reflections in episodic memory across trials and improve without weight updates. [[2]] (grounding 0.32, confidence 0.71)
- A single model generates an output, critiques it, and refines it iteratively with no extra training. [[3]] (grounding 0.32, confidence 0.70)

### What does empirical evidence say about does iterative self reflection improve?
_No new grounded claim here; see above or marked UNRESOLVED._

### What are the key trade-offs and failure modes for does iterative self reflection improve?
_No new grounded claim here; see above or marked UNRESOLVED._

### What reliability and failure-mode considerations apply to does iterative self reflection improve?
_No new grounded claim here; see above or marked UNRESOLVED._

## Contradictions

- **[OPEN]** on _What approaches exist for does iterative self reflection improve?_: Comparable evidence -> follow-up retrieval / human adjudication required
- **[OPEN]** on _What approaches exist for does iterative self reflection improve?_: Comparable evidence -> follow-up retrieval / human adjudication required

## Open questions / caveats

- 3 claim(s) dropped by the grounding gate.

## References

1. Asai et al. (2024). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. ICLR. arXiv:2310.11511.
2. Shinn et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. NeurIPS. arXiv:2303.11366.
3. Madaan et al. (2023). Self-Refine: Iterative Refinement with Self-Feedback. NeurIPS. arXiv:2303.17651.

---
_Every claim above resolves to a cited source above the grounding threshold; unresolved sub-questions are labelled, not answered._
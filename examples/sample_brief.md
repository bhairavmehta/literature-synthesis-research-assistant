# Technical Brief

**Question.** Should we adopt LLM-as-Judge for our evaluation pipeline, and under what calibration regime?

_Decision impact: high. Route: SYNC_REVIEW._

## Findings

### What approaches exist for llm judge evaluation pipeline calibration?
- This synthetic sample entry argues that LLM-as-Judge evaluation reliably matches human judgment and can replace human annotation in evaluation pipelines when a strong judge model is used for scoring calibration. [[1]] (grounding 0.61, confidence 0.98)
- This synthetic sample entry shows that LLM-as-Judge evaluation cannot match human judgment for high-stakes calibration and fails under position and verbosity bias, making it unreliable without human oversight. [[2]] (grounding 0.52, confidence 0.95)
- An LLM-as-Judge built from auto-generated evaluation steps and form-filling scores correlates with human judgment far better than overlap metrics on summarization. [[3]] (grounding 0.40, confidence 0.85)

### What does empirical evidence say about llm judge evaluation pipeline calibration?
- Strong LLM judges agree with human preferences at rates comparable to human-human agreement, but exhibit position, verbosity, and self-preference biases that must be mitigated through randomized and controlled evaluation. [[4]] (grounding 0.36, confidence 0.78)

### What are the key trade-offs and failure modes for llm judge evaluation pipeline calibration?
_No new grounded claim here; see above or marked UNRESOLVED._

### What reliability and failure-mode considerations apply to llm judge evaluation pipeline calibration?
_No new grounded claim here; see above or marked UNRESOLVED._

## Contradictions

- **[OPEN]** on _What approaches exist for llm judge evaluation pipeline calibration?_: Comparable evidence -> follow-up retrieval / human adjudication required
- **[RESOLVED]** on _What approaches exist for llm judge evaluation pipeline calibration?_: Adjudicated toward stronger evidence (Δ=0.12): This synthetic sample entry shows that LLM-as-Judge evaluation cannot match huma
- **[OPEN]** on _What does empirical evidence say about llm judge evaluation pipeline calibration?_: Comparable evidence -> follow-up retrieval / human adjudication required

## References

1. Synthetic Author A (2025). On the Sufficiency of LLM-as-Judge for Evaluation Pipelines. Synthetic Workshop. _(synthetic sample entry)_
2. Synthetic Author B (2025). Limits of LLM-as-Judge Under Bias. Synthetic Workshop. _(synthetic sample entry)_
3. Liu et al. (2023). G-Eval: NLG Evaluation Using GPT-4 with Better Human Alignment. EMNLP. arXiv:2303.16634.
4. Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS. arXiv:2306.05685.

---
_Every claim above resolves to a cited source above the grounding threshold; unresolved sub-questions are labelled, not answered._
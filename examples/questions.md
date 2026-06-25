# Sample research questions

These exercise different parts of the pipeline.

1. **Should we adopt LLM-as-Judge for our evaluation pipeline, and under what calibration regime?**
   _High impact — triggers synchronous HITL review; exercises the contradiction resolver (pro vs. con synthetic sources) and the injection scanner (poisoned source)._

2. **What retrieval strategy best reduces hallucinated citations in RAG?**
   _Exercises HyDE → rerank → CRAG and the grounding gate._

3. **When does iterative self-reflection improve an agent's reasoning?**
   _Low impact — typically routes AUTO_PASS / ASYNC_BATCH._

Run any of them:

```bash
python examples/demo.py --question "What retrieval strategy best reduces hallucinated citations in RAG?" --impact medium --trace
```

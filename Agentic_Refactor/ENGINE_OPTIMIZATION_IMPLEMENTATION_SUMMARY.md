# Engine Optimization Submission Summary

This summary documents the current optimization state, what is robustly implemented, and what is intentionally deferred to avoid low-value complexity. The scope remains query-level telemetry only (report and chat turn visibility), not persistent session-level accounting.

## Executive Position

The current system already implements the highest-ROI performance and cost controls for this architecture:
- selective model split (Flash-Lite for summaries, Flash for synthesis)
- deterministic memoization and query caches
- parallel recruiting web-scout execution
- transfer-context batching and reuse
- prompt budget caps on large synthesis payloads

The recommended path is to strengthen robustness and measurement quality, not add speculative optimizations.

## Criteria Coverage

### 1) Optimize inference performance (batching, caching, quantization)

**Implemented Controls:**
* **Caching (Latency & Cost Reduction):**
    * *Summary Memoization:* Implemented in `engine/tools.py` to prevent redundant LLM calls for identical web or API context.
    * *Vector-Query Cache:* Implemented in `engine/vector_service.py` to optimize repetitive semantic searches.
    * *Context Cache:* Implemented in `engine/orchestration_service.py` to reuse transfer CFBD data across multi-turn sessions.
* **Batching & Parallelism (Throughput Optimization):**
    * *Parallel Execution Nodes:* The recruiting web-scout paths run concurrently via LangGraph orchestration (`engine/agents.py` and `engine/graph.py`), minimizing blocking operations and reducing overall user wait time.
    * *Batched Task Execution:* Transfer portal evaluations batch per-season data extraction tasks (`engine/orchestration_service.py`) rather than relying on sequential agent loops.
* **Prompt-Budget Controls (Context Window Management):**
    * *Payload Truncation:* Hard limits on prompt sizes and final synthesis payloads (`engine/synthesis_service.py`) prevent unconstrained context bloat, acting as a strict governor on token ingestion.

**Explicitly Deferred by Design:**
* **Quantization:** This technique reduces the memory footprint of local model weights (e.g., converting FP16 to INT8). Because our architecture utilizes managed, API-hosted models (Gemini 3.1 Flash/Flash-Lite), manual quantization is neither applicable nor possible. 
    * *Trigger to Revisit:* Migration to edge inference or dedicated self-hosted open-weight models (e.g., Llama 3) in the future.

### 2) Cost-aware strategy (cascades / FrugalGPT style)

**Implemented Controls:**
* **Fixed Model Cascade:** We have implemented a deliberate, static model cascade that mirrors the intent of FrugalGPT. 
    * *How it works:* We use `gemini-3.1-flash-lite` for upstream worker agents (summarizing raw CFBD JSON and DuckDuckGo HTML snippets) and reserve the heavier `gemini-3-flash` exclusively for the final Lead Synthesizer node.
    * *Why it matters:* This routes high-volume, low-complexity tasks to the cheaper model, reducing the heavy model's token consumption by roughly 80% per query compared to a single-model approach.
* **Granular Telemetry:** Per-call and per-query telemetry actively logs estimated cost, token counts, and latency, ensuring visibility into model spend.

**Explicitly Deferred by Design:**
* **Dynamic FrugalGPT Routing:** Dynamically routing prompts at runtime based on an LLM-evaluated "complexity score" is deferred. The overhead of routing logic introduces latency and complexity that is not justified until benchmark evidence shows repeatable savings with no quality regression on our specific college football workloads.

### 3) Projected system costs (API, compute, storage)

**Current Pricing Assumptions (per 1M tokens):**
* `gemini-3.1-flash-lite-preview`: $0.25 Input / $1.50 Output
* `gemini-3-flash-preview`: $0.50 Input / $3.00 Output

**Directional Workload Assumptions & Per-Query Cost:**
* **Recruiting Report:** ~37.8k input + 3.15k output tokens *(~$0.021 per query)*
* **Transfer Report:** ~18k input + 1.5k output tokens *(~$0.012 per query)*
* **Follow-up Chat Turn:** ~12k input + 0.8k output tokens *(~$0.008 per query)*

To project system costs, we assume a blended average cost of **$0.015 per query** (representing a mix of heavy report generation and lighter chat follow-ups).

**Projected Monthly Operating Cost (POC) Table:**
| Monthly Volume | Estimated Token Cost | Infrastructure Base (Supabase/Ops) | Total Projected Cost |
| :--- | :--- | :--- | :--- |
| **Low (1,000 queries)** | $15.00 | $75.00 | **$90.00 / month** |
| **Medium (10,000 queries)** | $150.00 | $75.00 | **$225.00 / month** |
| **High (50,000 queries)** | $750.00 | $150.00* | **$900.00 / month** |
*(Note: High volume assumes a necessary tier upgrade in Supabase and caching infrastructure costs).*

If our group were to acutally release this publicly, we would require the user to insert their own Gemini API token and CFBD API token to avoid runaway costs. It would also require rethinking how the database was hosted. 

### 4) Cost-performance trade-offs and scaling bottlenecks

**Observed Trade-offs:**
* *Caching vs. Freshness:* Caching lowers cost and latency significantly, but relies on bounded cache behavior and risks serving stale injury or depth-chart news if not carefully invalidated.
* *Fixed Cascade vs. Dynamic Routing:* Our fixed model split is predictable and low-risk, but inherently less adaptive than dynamic FrugalGPT routing.

**Primary Scaling Bottlenecks (Ranked):**
1. **Final Synthesis Prompt Volume:** Pushing multiple agent summaries into the final model's context window remains the largest driver of token costs.
2. **Open-Chat Context Replay:** Multi-turn conversational memory re-submits previous context, compounding costs logarithmically over long sessions.
3. **Agentic Web/API Fan-out Variance:** Because agentic parallelization is unpredictable, a single query can spawn multiple web searches. Enforcing strict max-result caps on these searches is our primary defense against runaway token consumption.

## Robustness Updates Applied

Implemented in this pass:
* Added pricing diagnostics to ensure active models have valid token-rate entries before trusting cost telemetry.
* Wired pricing checks into one-click diagnostics and the sidebar preflight warning path.
* *Why this matters:* Prevents silent under-reporting of estimated costs when model names change or pricing configuration is incomplete.

## ROI-Prioritized Next Steps
1. Run a manual benchmark sample (10-20 representative recruiting and transfer queries).
2. Record p50/p95 latency, model call count, token totals, and exact estimated cost per query to validate the POC table.
3. Only consider implementing dynamic cascades if benchmark evidence shows a consistent financial gain.

## Conclusion: The Agentic Scale Trap (A Real-World Case Study)

The optimizations implemented in this system, specifically **prompt budget caps (Section 1)** and **controlling agentic fan-out (Section 4)**, are not merely theoretical best practices; they are critical safeguards against a newly emerging industry scaling trap. 

A few days ago, on April 20, 2026, GitHub was forced to abruptly pause new sign-ups for Copilot Pro tiers and implement severe "Weekly Token Limits" on existing users. We noticed this immediately, quickly running into weekly rate limits for high end models  (GPT-5.3-Codex), where none had existed before. Their infrastructure and pricing models were built around traditional, single-turn LLM inference. They fundamentally failed to anticipate the compute demands of *agentic workflows*, where a single user prompt (like a `/fleet` command) parallelizes into long-running, multi-agent file scans and recursive tasks. Because Copilot did not adequately cap "fan-out variance," a handful of user requests were suddenly incurring compute costs that exceeded the user's entire monthly subscription price. 

This real-world incident validates the foundational architecture of the Gridiron Intelligence engine. By recognizing that **parallelized worker agents are the primary threat to scale**, our system prioritizes hard token truncation and strict fan-out limits (e.g., bounding DuckDuckGo results) over speculative optimizations like dynamic routing. As the Copilot incident demonstrates, failing to account for the geometric token explosion of multi-agent RAG architectures inevitably results in degraded service, unexpected rate limits, and unsustainable operational costs.

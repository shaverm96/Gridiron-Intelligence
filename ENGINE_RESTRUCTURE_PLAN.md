## Plan: Engine Restructure in Manageable Portions

This is the execution index. Full detail is stored in ENGINE_RESTRUCTURE_FULL_PLAN.md in the same project directory.

**Portions**
1. Portion A: Baseline guardrails and parity checklist.
2. Portion B: Service-layer extraction from single-page app logic.
3. Portion C: Structured scouting page rebuild with rating-based player sorting.
4. Portion D: Structured engine fan-out/fan-in update with CFBD removed from structured path.
5. Portion E: Separate open-ended chat page with session-only memory.
6. Portion F: Navigation integration and shared diagnostics shell.
7. Portion G: Verification, hardening, and rollout handoff.

**Execution Order**
1. A -> B -> C
2. B -> D
3. D -> E
4. C + E -> F
5. A + B + C + D + E + F -> G

**Locked Decisions**
1. Player dropdown sort basis: recruiting rating only.
2. School dropdown behavior: target team context.
3. Chat memory scope: session-only.

**Scope Locks**
1. Included: Keep dropdown scouting flow and add separate open chat page.
2. Included: Structured report uses player web plus team web context.
3. Excluded: CFBD in structured report path.
4. Excluded: persistent cross-session memory.

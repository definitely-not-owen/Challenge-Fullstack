# LLM-Powered Clinical Trial Ranker (Demo Spec, Hybrid Approach)

## Overview
This demo ranker retrieves trials from BioMCP and produces ranked results per patient. It uses a **hybrid design**:
- **Deterministic filters** for hard eligibility constraints.
- **LLM scoring** for nuanced, qualitative judgment.
- **Latest frontier models** (GPT-5 / Claude 4.1 Sonnet / Gemini 2.5 flash) for high-quality outputs.

---

## Architecture

┌─────────────────┐
│  Trial Fetcher  │───┐
│   (BioMCP)      │   │
└─────────────────┘   │
                      ▼
             ┌─────────────────┐
             │ Deterministic   │
             │   Filters       │
             │  (rule-based)   │
             └─────────────────┘
                      │
                      ▼
             ┌─────────────────┐
             │   LLM Ranker    │
             │ (latest models) │
             └─────────────────┘
                      │
             ┌────────┴────────┐
             │                 │
      ┌──────▼──────┐   ┌─────▼─────────┐
      │  Scoring    │   │ Explainability│
      │ Engine      │   │   Module      │
      └─────────────┘   └───────────────┘

---

## Deterministic Filters (Pre-LLM)

Applied before sending trials to the LLM:

- **Age Cutoffs** (numeric comparison).
- **Gender Restrictions** (binary filter).
- **Trial Status** (exclude “Withdrawn,” “Completed”).
- **Geography** (filter out >X miles away, configurable).
- **Biomarker Hard Exclusions** (if explicitly listed in exclusion criteria text).

This reduces noise and ensures the LLM is only ranking trials where the patient is *potentially* eligible.

---

## LLM Scoring

### Model Choice
- Use **latest frontier models** (e.g., GPT-5, Claude 4.1 Sonnet, Gemini 2.5 flash).
- Optionally wrap with **Mixture-of-Experts**:
  - **Medical Expert Prompt** → eligibility nuance.
  - **Biomarker Specialist Prompt** → molecular matching.
  - **Patient Advocate Prompt** → quality-of-life/practicality.
  - **LLM Judge** (meta-model like Gemini 2.5 Pro) → adjudicates differences and produces final score.

This is **feasible** for a demo: you can parallelize multiple LLM calls, then consolidate. In production, cost/latency could be high, but for demos it’s impressive.

---

## Scoring Criteria (Hybridized)

**Eligibility (40 pts)**
- **Rule-based (deterministic):** age, gender, trial status.
- **LLM-assisted:** performance status, nuanced inclusion/exclusion.

**Biomarker Alignment (30 pts)**
- **Rule-based:** hard exclusions if biomarker explicitly mismatched.
- **LLM-assisted:** relevance of exploratory vs mandatory biomarkers.

**Clinical Appropriateness (20 pts)**
- **LLM-assisted:** trial phase vs disease stage, prior therapy logic.

**Practical Factors (10 pts)**
- **Rule-based:** geography threshold, open vs closed trial.
- **LLM-assisted:** subtle feasibility issues (travel burden, scheduling).

---

## Output Schema

Add a `deterministic_filters_applied` field for transparency.

{
  "trial_id": "NCT01234567",
  "total_score": 82,
  "subscores": {
    "eligibility": 35,
    "biomarker": 25,
    "clinical": 15,
    "practical": 7
  },
  "confidence": 0.84,
  "reasoning": "Patient meets inclusion; biomarkers aligned...",
  "key_matches": ["ER+ breast cancer", "CDK4/6 inhibitor"],
  "concerns": ["Trial requires ECOG 0–1"],
  "deterministic_filters_applied": {
    "age_ok": true,
    "gender_ok": true,
    "trial_open": true,
    "geography_ok": false
  }
}

---

## Evaluation Strategy (Demo-Friendly)

- **Synthetic expert labels:** useful for fast iteration.
- **Mixture-of-Experts + LLM Judge:** compare outputs from multiple perspectives, meta-model consolidates.
- **Stability checks:** rerun trials twice → measure consistency.

Even without a gold standard, this shows robustness and “reasoned” evaluation.

---

## Benefits of This Upgrade

1. **Safer demo:** obvious mismatches filtered out without LLM.
2. **Latest models:** results are more fluent, structured, and reliable.
3. **Mixture-of-Experts + Judge:** demonstrates sophistication — valuable in a demo pitch.
4. **Transparency:** deterministic filter logs show what the LLM didn’t even see, reinforcing trust.

---

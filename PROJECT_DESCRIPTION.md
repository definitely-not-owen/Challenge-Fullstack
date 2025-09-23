# Full-Stack Engineer Take-Home: Clinical Trial Matching System

## Challenge Overview
Build a clinical trial matcher. We want to see:
1. **BioMCP integration** to fetch real trials
2. **LLM-powered ranking** (be clever here!)
3. **Evaluation suite** you're proud of (get creative: synthetic data, LLM-as-judge, whatever works)

**Time:** ~3 hours. Ship fast, be scrappy, show your thinking.

## What You Need to Build, in no particular order

### 1. **Trial Fetcher**
- Use BioMCP to get real clinical trials.
- Fetch cleverly.

### 2. **LLM Ranker** 
- Use any LLM(s) to rank trials.
- Rank cleverly.

### 3. **Evaluation Suite** ⭐ THIS IS KEY
We want to see how you think about evaluation.
- Can be hard-coded, synthetic data, real-life data, LLM-as-judge, reward models, RLHF/RLAIF, etc. Get creative and work fast.
- Your main prerogative: Stand up an eval suite you're proud of!

**Just make it run:**
```bash
python src/match.py --patient_id P002
# Output: Ranked NCT IDs with scores

python tests/test_matching.py  
# Output: Your eval metrics, whatever you choose
```

## Example codebase structure

```
name-your-solution/
├── patients.csv          # We provide this
├── src/
│   └── match.py         # Your matcher (can split into modules if you want)
└── tests/
    └── eval.py  # Your eval suite (can split into modules if you want)
```

Structure however you'd like, free to add more files.

## Technical Details

### BioMCP SDK Usage
The BioMCP SDK provides **live clinical trial data**.
Think about what you'd use to filter.

**Got a specialty?** Show it off! Whether it's RL, RAG, fine-tuning, UI/UX, or advanced prompting - add your flair.

## What Makes a Great Submission

**We care about:**
- It works (can run your commands and get results)
- Understanding of strengths and limits of LLMs
- Thoughtful evaluation (show you understand the problem)
- Rate of learning, Rate of execution

**We don't care about:**
- Perfect code architecture
- 100% test coverage
- Supporting every edge case

## Quick Start

```bash
# Install basics
pip install biomcp pandas openai  # or whatever you need

# Start here:
1. Get BioMCP working - fetch some trials
2. Hook up an LLM - rank them somehow  
3. Build an eval - prove it works, and iterate

# Tip: If you'd like, focus on one cancer to start. Breast cancer works well.
```

## Submission Instructions

1. **Complete your solution** in a private GitHub repo
2. **Add bryan as a collaborator**
3. **Include a brief README** explaining:
   - How to run your solution
   - Your approach and key decisions
   - Any TODOs or limitations
4. **Email the repo link** to bryan@radicalhealth.ai & simone@radicalhealth.ai
5. **Book a 45-minute walkthrough** to discuss your solution

## Final Thoughts

This is deliberately open-ended. We want to see:
- How you approach ambiguous problems
- Your instincts about what matters
- How you balance speed vs quality
- What you consider "good enough" for a 3-hour sprint

Remember: **We're a startup.** Show us you can ship fast, think clearly, and build things that work.

Good luck! 

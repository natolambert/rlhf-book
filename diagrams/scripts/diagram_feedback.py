#!/usr/bin/env python3
"""
Diagram feedback loop using Gemini API.

Uses a council of reviewers (Gemini + local analysis) to iterate on diagram quality.
Passes relevant chapter content + math + generated diagrams for review.

Usage:
    GEMINI_API_KEY=... uv run python scripts/diagram_feedback.py

    # Or with env file
    export GEMINI_API_KEY=...
    uv run python scripts/diagram_feedback.py --iterations 2
"""

import argparse
import base64
import os
from pathlib import Path
from dataclasses import dataclass

import google.generativeai as genai

# Chapter 7 relevant sections for each diagram type
CHAPTER_CONTEXT = {
    "pref_rm": {
        "title": "Bradley-Terry Preference Reward Model",
        "section": """
The canonical implementation of a reward model is derived from the Bradley-Terry model of preference.
A Bradley-Terry model of preferences defines the probability that, in a pairwise comparison between two items i and j, a judge prefers i over j:

P(i > j) = p_i / (p_i + p_j)

Reparametrized with unbounded scores where p_i = e^{r_i}:

P(i > j) = e^{r_i} / (e^{r_i} + e^{r_j}) = σ(r_i - r_j)

The loss function to train a reward model:
L(θ) = -log(σ(r_θ(y_c | x) - r_θ(y_r | x)))

Where y_c is the chosen completion and y_r is the rejected completion.

Architecture: A small linear head appended to a language model that outputs a single scalar at the EOS/last token.
The model outputs the probability that the piece of text is chosen as a single logit.
""",
        "key_points": [
            "Sequence-level scalar output at EOS token",
            "Pairwise contrastive loss (BT loss)",
            "Only differences in scores matter",
            "Linear head on LM trunk",
        ],
    },
    "orm": {
        "title": "Outcome Reward Model (ORM)",
        "section": """
For reasoning heavy tasks, one can use an Outcome Reward Model (ORM).
Training data: problem statement/prompt x, and two completions where one is correct (y_c) and one incorrect (y_ic).

The loss is a per-token cross-entropy:
L_CE(θ) = -E_{(s,r)~D}[r·log p_θ(s) + (1-r)·log(1-p_θ(s))]

Where r ∈ {0,1} is a binary label (1=correct, 0=incorrect), and p_θ(s) is the predicted probability of correctness.

Architecture: Language modeling head that predicts two classes per token (1 for correct, 0 for incorrect).
The important intuition: an ORM outputs a probability of correctness at every token in the sequence.
Prompt tokens are masked (labels = -100), completion tokens are supervised.
""",
        "key_points": [
            "Per-token correctness prediction",
            "Binary cross-entropy loss",
            "Prompt tokens masked, completion supervised",
            "Outputs probability of correctness at each token",
        ],
    },
    "prm": {
        "title": "Process Reward Model (PRM)",
        "section": """
Process Reward Models (PRMs) are trained to output scores at every *step* in a chain-of-thought reasoning process.
They differ from standard RMs (score at EOS) and ORMs (score at every token).

Binary-labeled PRM loss (per-step cross-entropy):
L_PRM(θ) = -E_{(x,s)~D}[Σ_{i=1}^{K} y_{s_i}·log r_θ(s_i|x) + (1-y_{s_i})·log(1-r_θ(s_i|x))]

Where:
- s is a chain-of-thought with K annotated steps
- y_{s_i} ∈ {0,1} denotes whether the i-th step is correct
- r_θ(s_i|x) is the PRM's predicted probability that step s_i is valid

Labels: -1 for incorrect, 0 for neutral, 1 for correct (at step boundaries only).
Step boundaries typically at double newlines or special tokens.
Only tokens at step boundaries have labels (others masked as -100).
""",
        "key_points": [
            "Step-boundary supervision only",
            "3-class prediction: incorrect/neutral/correct",
            "Non-boundary tokens masked",
            "Used for reasoning/CoT evaluation",
        ],
    },
    "gen_rm": {
        "title": "Generative Reward Model (LLM-as-Judge)",
        "section": """
With the cost of preference data, a large research area emerged to use existing language models as a judge.
The core idea: prompt a language model with instructions on how to judge, a prompt, and two completions.

No separate reward head needed - uses the LLM's generation capabilities.
Output: natural language verdict that is then parsed to extract a score or preference.

Example prompt structure:
- System: "Please act as an impartial judge..."
- User Question: {question}
- Assistant A's Answer: {answer_a}
- Assistant B's Answer: {answer_b}
- Output format: "[[A]]" if A is better, "[[B]]" if B is better, "[[C]]" for tie

Can be used zero-shot, few-shot, or fine-tuned specifically as a judge.
Common trick: use sampling temperature of 0 to reduce variance.
""",
        "key_points": [
            "No reward head - uses generation",
            "Natural language verdict parsed to score",
            "Zero/few-shot or fine-tuned",
            "Prompt engineering for judgment criteria",
        ],
    },
}

# Summary table from chapter for context
SUMMARY_TABLE = """
| Model Class | What They Predict | How They Are Trained | LM structure |
|------------|------------------|---------------------|--------------|
| **Reward Models** | Quality of text via probability of chosen response at EOS token | Contrastive loss between pairwise comparisons | Regression/classification head on LM features |
| **Outcome Reward Models** | Probability that an answer is correct per-token | Labeled outcome pairs (success/failure) | LM head per-token cross-entropy |
| **Process Reward Models** | Score for intermediate steps at end of reasoning steps | Intermediate feedback or stepwise annotations | LM head per reasoning step, predicts 3 classes |
| **Value Functions** | Expected return given current state | Regression to each point in sequence | Classification with output per-token |
"""


@dataclass
class DiagramReview:
    """Result of a diagram review."""
    diagram_name: str
    analysis: str
    findings_table: str
    summary: str
    suggestions: list[str]


def load_image_as_base64(image_path: Path) -> str:
    """Load an image and convert to base64."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def create_review_prompt(diagram_type: str, is_token_strip: bool = True) -> str:
    """Create the review prompt for Gemini."""
    context = CHAPTER_CONTEXT.get(diagram_type, CHAPTER_CONTEXT["pref_rm"])

    diagram_format = "token strip visualization" if is_token_strip else "pipeline/flowchart diagram"

    prompt = f"""YOU: are an academic reviewer specializing in machine learning pedagogy and technical illustration.
You have expertise in RLHF (Reinforcement Learning from Human Feedback), reward modeling, and creating clear educational diagrams.

INPUT: A {diagram_format} diagram for "{context['title']}" from an RLHF textbook chapter on Reward Modeling.

CONTEXT FROM CHAPTER:
{context['section']}

KEY POINTS THE DIAGRAM SHOULD CONVEY:
{chr(10).join(f"- {p}" for p in context['key_points'])}

SUMMARY TABLE FOR REFERENCE:
{SUMMARY_TABLE}

OUTPUT: Please provide your analysis in the following structure:

## Diagram Analysis

[Detailed analysis of what the diagram shows, what it gets right, and what could be improved]

## Extracted Information

| Element | What It Shows | Accuracy | Clarity |
|---------|--------------|----------|---------|
[Extract each visual element and evaluate it]

## Summary

[Three sentence summary: (1) Overall assessment, (2) Main strength, (3) Priority improvement]

## Specific Suggestions

[Numbered list of concrete, actionable improvements]
"""
    return prompt


def review_diagram_with_gemini(
    api_key: str,
    image_path: Path,
    diagram_type: str,
    is_token_strip: bool = True,
) -> DiagramReview:
    """Send diagram to Gemini for review."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Load image
    image_data = load_image_as_base64(image_path)

    # Create prompt
    prompt = create_review_prompt(diagram_type, is_token_strip)

    # Send to Gemini
    response = model.generate_content([
        prompt,
        {
            "mime_type": "image/png",
            "data": image_data,
        }
    ])

    # Parse response
    text = response.text

    # Extract sections (simple parsing)
    analysis = ""
    findings_table = ""
    summary = ""
    suggestions = []

    sections = text.split("## ")
    for section in sections:
        if section.startswith("Diagram Analysis"):
            analysis = section.replace("Diagram Analysis", "").strip()
        elif section.startswith("Extracted Information"):
            findings_table = section.replace("Extracted Information", "").strip()
        elif section.startswith("Summary"):
            summary = section.replace("Summary", "").strip()
        elif section.startswith("Specific Suggestions"):
            sugg_text = section.replace("Specific Suggestions", "").strip()
            # Parse numbered list
            for line in sugg_text.split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    suggestions.append(line.lstrip("0123456789.-) "))

    return DiagramReview(
        diagram_name=image_path.name,
        analysis=analysis,
        findings_table=findings_table,
        summary=summary,
        suggestions=suggestions,
    )


def local_analysis(image_path: Path, diagram_type: str) -> str:
    """Provide local analysis based on diagram specs."""
    context = CHAPTER_CONTEXT.get(diagram_type, {})

    analysis = f"""## Local Analysis for {diagram_type}

### Expected Elements
{chr(10).join(f"- {p}" for p in context.get('key_points', []))}

### Diagram File
- Path: {image_path}
- Exists: {image_path.exists()}
- Size: {image_path.stat().st_size if image_path.exists() else 'N/A'} bytes

### Checklist
- [ ] Shows correct data flow
- [ ] Labels match chapter terminology
- [ ] Supervision pattern is clear
- [ ] Loss function is represented
- [ ] Architecture distinction is visible
"""
    return analysis


def run_feedback_loop(
    api_key: str,
    generated_dir: Path,
    output_dir: Path,
    iterations: int = 2,
) -> None:
    """Run the feedback loop for all diagrams."""

    # Token strip diagrams
    token_strips = [
        ("pref_rm_tokens.png", "pref_rm"),
        ("orm_tokens.png", "orm"),
        ("prm_tokens.png", "prm"),
        ("value_fn_tokens.png", "pref_rm"),  # value fn uses similar context
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    all_feedback = []

    for iteration in range(1, iterations + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}")
        print(f"{'='*60}")

        for filename, diagram_type in token_strips:
            image_path = generated_dir / filename

            if not image_path.exists():
                print(f"\n⚠️  Skipping {filename} - file not found")
                continue

            print(f"\n📊 Reviewing: {filename}")
            print(f"   Type: {CHAPTER_CONTEXT[diagram_type]['title']}")

            # Local analysis
            local = local_analysis(image_path, diagram_type)

            # Gemini review
            try:
                review = review_diagram_with_gemini(
                    api_key, image_path, diagram_type, is_token_strip=True
                )

                feedback_md = f"""# Feedback: {filename}
## Iteration {iteration}

{local}

## Gemini Analysis

{review.analysis}

## Findings Table

{review.findings_table}

## Summary

{review.summary}

## Suggestions for Improvement

{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(review.suggestions))}

---
"""
                all_feedback.append(feedback_md)

                # Print summary
                print(f"\n   Summary: {review.summary[:200]}...")
                if review.suggestions:
                    print(f"   Top suggestion: {review.suggestions[0]}")

            except Exception as e:
                print(f"   ❌ Error: {e}")
                all_feedback.append(f"# Feedback: {filename}\n\nError: {e}\n\n---\n")

    # Write combined feedback
    feedback_path = output_dir / "diagram_feedback.md"
    with open(feedback_path, "w") as f:
        f.write("# Diagram Feedback Report\n\n")
        f.write("Generated by Gemini + local analysis council.\n\n")
        f.write("---\n\n")
        f.write("\n".join(all_feedback))

    print(f"\n✅ Feedback written to: {feedback_path}")


def main():
    parser = argparse.ArgumentParser(description="Diagram feedback loop with Gemini")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GEMINI_API_KEY"),
        help="Gemini API key (or set GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=Path(__file__).parent.parent / "generated",
        help="Directory with generated diagrams",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "feedback",
        help="Directory for feedback output",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2,
        help="Number of feedback iterations",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("❌ Error: GEMINI_API_KEY not set")
        print("   Set via --api-key or GEMINI_API_KEY environment variable")
        return

    run_feedback_loop(
        api_key=args.api_key,
        generated_dir=args.generated_dir,
        output_dir=args.output_dir,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()

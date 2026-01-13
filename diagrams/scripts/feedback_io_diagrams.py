#!/usr/bin/env python3
"""
Get Gemini feedback on I/O diagrams for reward models.
"""

import base64
import os
from pathlib import Path

import google.generativeai as genai

CHAPTER_CONTEXT = """
From Chapter 7 - Reward Modeling:

## Preference RM (Bradley-Terry)
- Input: (prompt x, completion y)
- Architecture: LM trunk + linear scalar head at EOS token
- Output: Single scalar r(x,y) representing quality
- Loss: L = -log σ(r_c - r_r) for pairwise comparison
- Usage: Rank completions, provide reward signal for RL

## Outcome Reward Model (ORM)
- Input: (prompt x, completion y)
- Architecture: LM trunk + per-token classifier head
- Output: p(correct | token) for each completion token
- Loss: BCE per token, prompt tokens masked (label=-100)
- Usage: Aggregate per-token probs (mean/last) for verification

## Process Reward Model (PRM)
- Input: (prompt x, CoT steps s1...sK)
- Architecture: LM trunk + 3-class head at step boundaries
- Output: Score at each step boundary (correct/neutral/incorrect)
- Loss: CE only at boundary tokens (others masked)
- Usage: Aggregate step scores (min/product), guide search

## Generative RM (LLM-as-Judge)
- Input: Judge prompt with (x, completions, rubric)
- Architecture: Full LLM (GPT-4, Claude, etc.) - no reward head
- Output: Natural language verdict → parsed to score/preference
- Loss: Standard LM loss if fine-tuned, often used zero-shot
- Usage: Evaluation, can explain reasoning
"""

REVIEW_PROMPT = """YOU: are an academic reviewer specializing in machine learning pedagogy,
technical illustration, and RLHF systems. You have deep expertise in reward modeling.

INPUT: An input-output flow diagram showing inference for a reward model type.
These are rough mockups intended for iteration before professional artist handoff.

CONTEXT FROM CHAPTER:
{context}

OUTPUT: Provide your analysis in this structure:

## Visual Assessment
[Is the flow clear? Are the boxes/arrows readable? Color coding effective?]

## Technical Accuracy
[Does the diagram correctly represent the inference process? Any errors?]

## Information Completeness
| Element | Present? | Accurate? | Suggestion |
|---------|----------|-----------|------------|
| Input specification | | | |
| Model architecture | | | |
| Processing steps | | | |
| Output format | | | |
| Key notes | | | |

## Summary
[3 sentences: (1) Overall quality, (2) Main strength, (3) Priority fix]

## Top 3 Improvements
[Numbered list of specific, actionable changes]
"""


def review_diagram(api_key: str, image_path: Path, diagram_name: str) -> str:
    """Get Gemini feedback on a diagram."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    prompt = REVIEW_PROMPT.format(context=CHAPTER_CONTEXT)

    response = model.generate_content([
        f"Reviewing: {diagram_name}\n\n{prompt}",
        {"mime_type": "image/png", "data": image_data},
    ])

    return response.text


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY environment variable")
        return

    generated_dir = Path(__file__).parent.parent / "generated"

    io_diagrams = [
        ("pref_rm_io.png", "Preference RM I/O"),
        ("orm_io.png", "Outcome RM I/O"),
        ("prm_io.png", "Process RM I/O"),
        ("gen_rm_io.png", "Generative RM I/O"),
    ]

    all_feedback = ["# I/O Diagram Feedback\n"]

    for filename, title in io_diagrams:
        path = generated_dir / filename
        if not path.exists():
            print(f"Skipping {filename} - not found")
            continue

        print(f"\n📊 Reviewing: {title}...")
        try:
            feedback = review_diagram(api_key, path, title)
            all_feedback.append(f"\n## {title}\n\n{feedback}\n\n---\n")

            # Print summary
            lines = feedback.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("## Summary"):
                    summary = "\n".join(lines[i+1:i+4]).strip()
                    print(f"   {summary[:150]}...")
                    break
        except Exception as e:
            print(f"   Error: {e}")
            all_feedback.append(f"\n## {title}\n\nError: {e}\n\n---\n")

    # Write feedback
    output_path = Path(__file__).parent.parent / "feedback" / "io_diagram_feedback.md"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(all_feedback))

    print(f"\n✅ Feedback saved to: {output_path}")


if __name__ == "__main__":
    main()

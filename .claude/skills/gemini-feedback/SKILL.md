---
name: gemini-feedback
description: Get feedback from Gemini API on a diagram image for textbook quality review.
allowed-tools: Bash(uv run python:*), Read
---

# Gemini Feedback

Get feedback from Gemini API on a diagram image.

## Usage

```
/gemini-feedback <path-to-image> [context]
```

## Instructions

When this command is invoked:

1. Read the image file at the provided path
2. Call the Gemini API to get feedback on the diagram
3. Use this Python snippet to call Gemini:

```python
import base64
import os
import google.generativeai as genai

genai.configure(api_key=os.environ['GEMINI_API_KEY'])
model = genai.GenerativeModel('gemini-2.0-flash')

with open('<IMAGE_PATH>', 'rb') as f:
    img = base64.standard_b64encode(f.read()).decode()

prompt = '''Review this diagram for a textbook. Be concise and specific.

Context: <CONTEXT>

Please provide:
1. Overall assessment (1-2 sentences)
2. Specific issues to fix (be detailed about visual problems like alignment, overlapping, etc.)
3. Suggestions for improvement
'''

response = model.generate_content([prompt, {'mime_type': 'image/png', 'data': img}])
print(response.text)
```

4. Replace `<IMAGE_PATH>` with the actual path provided
5. Replace `<CONTEXT>` with any context provided, or use "Technical diagram for ML/AI textbook"
6. Ensure GEMINI_API_KEY is set in environment
7. Run using: `uv run python -c "..."`
8. Present the feedback to the user

## Example

```
/gemini-feedback diagrams/generated/png/tool_use_generation.png "Diagram showing tool use interleaving in LLM generation"
```

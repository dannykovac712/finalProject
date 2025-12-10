import json
import os
import time
import re
from openai import OpenAI


client = OpenAI()


TOPICS = [
    "real analysis",
    "linear algebra",
    "probability theory",
    "topology",
    "abstract algebra",
    "multivariable calculus",
    "optimization",
    "number theory",
]


system_prompt = """
You generate training data for a model that extracts structured information
from mathematical theorems.

For each example, produce:
- "theorem": a short but realistic theorem statement (plain text, not LaTeX)
- "struct": a JSON object with fields:
    - type: e.g., "theorem", "lemma", "proposition", "corollary"
    - id: a short identifier like "thm:xyz123"
    - name: optional short name (string, may be empty)
    - assumptions: list of short strings
    - conclusion: one short string

You must return a SINGLE JSON object with exactly this shape:
{
  "examples": [
    {
      "theorem": "...",
      "struct": {
        "type": "...",
        "id": "...",
        "name": "...",
        "assumptions": ["..."],
        "conclusion": "...",
      }
    },
    ...
  ]
}

Return ONLY JSON. No explanations, no comments, no surrounding text.
"""


OUTPUT_FILE = "data.jsonl"
TARGET_EXAMPLES = 300
EXAMPLES_PER_BATCH = 10

seen_theorems = set()
records = []

def canonicalize(text: str) -> str:
    """
    Stronger normalization for duplicate detection:
    - lowercase
    - strip punctuation
    - normalize whitespace
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text) 
    text = re.sub(r"\s+", " ", text)
    return text.strip()

while len(records) < TARGET_EXAMPLES:
    print(f"\nRequesting a new batch... currently have {len(records)} examples")

    batch_index = len(records) // EXAMPLES_PER_BATCH
    topic = TOPICS[batch_index % len(TOPICS)]
    print(f"🔎 Using topic: {topic}")

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"""
Generate {EXAMPLES_PER_BATCH} diverse undergraduate-level math statements
specifically from **{topic}**.

Include a mix of the following types where appropriate:
- theorem
- lemma
- proposition
- corollary

Theorems should cover different assumptions and conclusions.
Do NOT repeat theorems within this batch.

Remember:
- Output a SINGLE JSON object with key "examples"
- DO NOT include any text that is not valid JSON.
""",
            },
        ],
    )

    raw_text = resp.output[0].content[0].text

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        print(" Failed to parse JSON. Raw response was:")
        print(raw_text[:500], "...")
        print("Skipping this batch and continuing...\n")
        time.sleep(1)
        continue

    if "examples" not in data or not isinstance(data["examples"], list):
        print("Parsed JSON but 'examples' key is missing or not a list. Got:")
        print(data)
        print("Skipping this batch and continuing...\n")
        time.sleep(1)
        continue

    batch_added = 0
    for ex in data["examples"]:
        if "theorem" not in ex or "struct" not in ex:
            continue

        theorem = ex["theorem"]
        norm = canonicalize(theorem)

        if norm in seen_theorems:
            continue

        seen_theorems.add(norm)

        record = {
            "input": theorem,
            "output": json.dumps(ex["struct"], ensure_ascii=False),
        }

        records.append(record)
        batch_added += 1
        print(f" Added {len(records)} unique examples")

        if len(records) >= TARGET_EXAMPLES:
            break

    print(f"Batch done, added {batch_added} new examples.")


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\n Finished with {len(records)} unique examples written to {OUTPUT_FILE}")

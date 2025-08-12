#!/usr/bin/env python3
"""
newsletter_maker.py — 3-call AI newsletter generator

Usage:
  pip install openai python-dotenv
  export OPENAI_API_KEY="sk-..."
  python newsletter_maker.py articles/a1.md articles/a2.md articles/a3.md \
      --date "August 12, 2025" --output newsletter.md

What it does (exactly 3 LLM calls):
  1) TLDR + per-article news blurbs  (1 call)
  2) Subject + preheader             (1 call; input = TLDR)
  3) Prompt of the week              (1 call; input = TLDR + headlines)

Default model: gpt-4o-mini (override with --model or $NEWSLETTER_MODEL).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI  # pip install openai

# ---------- Config ----------
DEFAULT_MODEL = os.getenv("NEWSLETTER_MODEL", "gpt-4o-mini")
MAX_CHARS_PER_ARTICLE = int(os.getenv("MAX_CHARS_PER_ARTICLE", "12000"))
SUBJECT_CHAR_LIMIT = 75
PREHEADER_CHAR_LIMIT = 110


def read_text(path: Path, limit: int) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[:limit].strip()


def make_client() -> OpenAI:
    # OPENAI_API_KEY is read automatically by the SDK if present in env
    return OpenAI()


def call_structured_json(
    client: OpenAI,
    *,
    model: str,
    instructions: str,
    user_input: str,
    schema_name: str,
    schema: Dict[str, Any],
    temperature: float = 0.5,
) -> Dict[str, Any]:
    """One-shot helper for structured outputs via Responses API."""
    resp = client.responses.create(
        model=model,
        temperature=temperature,
        instructions=instructions,
        input=user_input,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema,
                "strict": True,
            },
        },
    )
    # For structured outputs, the model returns a JSON string in output_text.
    return json.loads(resp.output_text)


# ---------- Prompts & Schemas ----------

TLDR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["tldr_bullets", "news"],
    "properties": {
        "tldr_bullets": {
            "type": "array",
            "minItems": 3,
            "maxItems": 5,
            "items": {"type": "string", "maxLength": 180},
        },
        "news": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["headline", "summary"],
                "properties": {
                    "headline": {"type": "string", "maxLength": 100},
                    "summary": {"type": "string", "maxLength": 800},
                },
            },
        },
    },
}

TLDR_INSTRUCTIONS = """You are a precise editorial assistant.
Given exactly three articles, produce:
1) `tldr_bullets`: 3–5 concise bullets (<= 180 chars each) that capture the *whole* week.
2) `news`: exactly three entries. For each, write:
   - `headline`: a short, neutral headline (<= 100 chars).
   - `summary`: 2–5 tight sentences, factual and hype-free.

Rules:
- Use only what’s in the articles; no external facts.
- Prefer present tense and concrete details.
- No emojis, no markdown formatting, no marketing fluff.
- Output must be strict JSON that matches the provided schema.
"""

SUBJECT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["subject", "preheader"],
    "properties": {
        "subject": {"type": "string", "maxLength": SUBJECT_CHAR_LIMIT},
        "preheader": {"type": "string", "maxLength": PREHEADER_CHAR_LIMIT},
    },
}

SUBJECT_INSTRUCTIONS = f"""You are an email subject-line expert.
Given a newsletter TL;DR, write:
- `subject`: <= {SUBJECT_CHAR_LIMIT} chars, clear, specific, non-clickbait.
- `preheader`: <= {PREHEADER_CHAR_LIMIT} chars, complements the subject.

Rules:
- Summarize the week’s value proposition. Avoid salesy language.
- No ALL CAPS, no emojis, no brackets like [NEWS].
- Output strict JSON only.
"""

POTW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "scenario", "task", "instruction"],
    "properties": {
        "title": {"type": "string", "maxLength": 120},
        "scenario": {"type": "string", "maxLength": 400},
        "task": {"type": "string", "maxLength": 400},
        "instruction": {"type": "string", "maxLength": 600},
    },
}

POTW_INSTRUCTIONS = """You design a weekly, immediately-useful LLM prompt.
Using the TL;DR and headlines, create a prompt-of-the-week with fields:
- `title`: punchy topic for practitioners (<=120 chars)
- `scenario`: a concrete work scenario
- `task`: what the user asks the model to produce
- `instruction`: constraints & style so outputs are high quality

Tone: practical, specific, and just a tad creative.
No emojis. Output strict JSON only.
"""

# ---------- Rendering ----------

HEADER = "AI Newsletter – Your weekly summary of AI news, Resources and Prompts"


def render_markdown(
    date_str: str,
    tldr: Dict[str, Any],
    subject_block: Dict[str, str],
    potw: Dict[str, str],
) -> str:
    lines: List[str] = []
    lines += ["Header", HEADER, f"Date: {date_str}", ""]
    lines += ["Summary TL;DR"]
    for b in tldr["tldr_bullets"]:
        lines.append(f"- {b}")
    lines += ["", "AI News"]
    for item in tldr["news"]:
        lines += [item["headline"], item["summary"], ""]
    lines += [
        "Prompt of the Week",
        f"Title: {potw['title']}",
        f"Scenario: {potw['scenario']}",
        f"Task: {potw['task']}",
        f"Instruction: {potw['instruction']}",
        "",
        f"(Subject) {subject_block['subject']}",
        f"(Preheader) {subject_block['preheader']}",
        "",
    ]
    return "\n".join(lines)


# ---------- Main pipeline ----------

def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Generate a weekly AI newsletter with 3 LLM calls.")
    parser.add_argument("articles", nargs=3, type=Path, help="Paths to the three article files (txt/md)")
    parser.add_argument("--date", default=dt.date.today().strftime("%B %d, %Y"), help="Human-readable date")
    parser.add_argument("-o", "--output", type=Path, default=Path("newsletter.md"), help="Output Markdown file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name (default from env NEWSLETTER_MODEL or gpt-4o-mini)")
    parser.add_argument("--temp", type=float, default=0.5, help="Generation temperature (default 0.5)")
    args = parser.parse_args(argv)

    # Read input articles (lightly truncated to control token use)
    raw_articles = [
        read_text(p, MAX_CHARS_PER_ARTICLE) for p in args.articles
    ]
    article_packet = "\n\n".join(
        f"ARTICLE {i+1}:\n{txt}" for i, txt in enumerate(raw_articles)
    )

    client = make_client()

    # 1) TLDR (+ per-article blurbs) — single call
    tldr_payload = call_structured_json(
        client,
        model=args.model,
        instructions=TLDR_INSTRUCTIONS,
        user_input=article_packet,
        schema_name="NewsletterTLDR",
        schema=TLDR_SCHEMA,
        temperature=args.temp,
    )

    # 2) Subject (+ preheader) — single call; input = TLDR only
    tl_bullets_str = "\n".join(f"- {b}" for b in tldr_payload["tldr_bullets"])
    subject_payload = call_structured_json(
        client,
        model=args.model,
        instructions=SUBJECT_INSTRUCTIONS,
        user_input=f"TL;DR bullets:\n{tl_bullets_str}",
        schema_name="SubjectSchema",
        schema=SUBJECT_SCHEMA,
        temperature=0.4 if args.temp >= 0.4 else args.temp,
    )

    # 3) Prompt of the Week — single call; input = TLDR + headlines
    headlines = "\n".join(f"- {n['headline']}" for n in tldr_payload["news"])
    potw_input = f"TL;DR:\n{tl_bullets_str}\n\nHeadlines:\n{headlines}"
    potw_payload = call_structured_json(
        client,
        model=args.model,
        instructions=POTW_INSTRUCTIONS,
        user_input=potw_input,
        schema_name="PromptOfTheWeek",
        schema=POTW_SCHEMA,
        temperature=0.6,
    )

    # Render final markdown
    md = render_markdown(args.date, tldr_payload, subject_payload, potw_payload)
    args.output.write_text(md, encoding="utf-8")
    print(f"Wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""
auto_newsletter.py — Fully automatic weekly AI newsletter generator.

What it does:
  - Gathers fresh AI news from Google News RSS queries and curated RSS feeds
  - Ranks + deduplicates items and extracts article text
  - Makes exactly 3 OpenAI Responses API calls:
       1) TL;DR bullets (3–5) + 3 “AI News” blurbs from the chosen articles
       2) Subject + preheader (from the TL;DR)
       3) Prompt of the Week (from TL;DR + headlines)
  - Renders a clean Markdown newsletter and lists sources

Quick start:
  pip install -r requirements.txt
  export OPENAI_API_KEY="sk-..."
  python auto_newsletter.py --output newsletters/$(date +%Y-%m-%d).md

Config knobs:
  --days 7                # time window for “fresh” items
  --max-candidates 40     # how many items to score before narrowing
  --model gpt-4o-mini     # or override via $NEWSLETTER_MODEL
  --date "August 12, 2025"

This script uses:
  - OpenAI Responses API with structured JSON outputs (json_schema)
  - feedparser for RSS
  - trafilatura for article text extraction
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import feedparser           # RSS parsing
import trafilatura          # article fetching + extraction
from dateutil import tz     # timezone helpers
from openai import OpenAI   # OpenAI SDK
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------------------- Config ----------------------

DEFAULT_MODEL = os.getenv("NEWSLETTER_MODEL", "gpt-4o-mini")
SUBJECT_CHAR_LIMIT = 75
PREHEADER_CHAR_LIMIT = 110
MAX_CHARS_PER_ARTICLE = int(os.getenv("MAX_CHARS_PER_ARTICLE", "12000"))
NEWSLETTER_TITLE = "AI Newsletter – Your weekly summary of AI news, Resources and Prompts"

GOOGLE_NEWS_PARAMS = "hl=en-US&gl=US&ceid=US:en"  # language/region; widely used pattern
GOOGLE_NEWS_QUERIES = [
    # broad + model families; keep short so we don’t over-filter
    '("artificial intelligence" OR "generative AI" OR "large language model")',
    '(GPT OR "ChatGPT" OR "GPT-5" OR "o1" OR "o4")',
    '(Gemini OR "Google DeepMind" OR "world model" OR "Genie 3")',
    '(Claude OR Anthropic OR Opus)',
    '(Llama OR Qwen OR Mixtral OR Mistral)',
    '("open weights" OR "AI Act" OR "AI safety" OR "compute")',
]

# High-signal first-party sources (handful, can extend)
CURATED_RSS = [
    # If a feed dies, the script just skips it.
    "https://ai.googleblog.com/feeds/posts/default?alt=rss",  # Google AI Blog
    "https://deepmind.google/discover/blog/feed/basic/",      # DeepMind Blog feed
    "https://www.theverge.com/ai-artificial-intelligence/rss/index.xml",  # Verge AI section
    "https://arstechnica.com/ai/feed/",                       # Ars Technica AI
    "https://news.mit.edu/topic/artificial-intelligence2-rss",# MIT News AI topic
]

# Keyword weights used in ranking (title + summary)
KEYWORD_WEIGHTS = {
    # Companies / labs / families
    "openai": 5, "gpt": 5, "chatgpt": 4, "gpt-5": 7, "o1": 5, "o4": 5,
    "google": 3, "deepmind": 5, "gemini": 5, "genie": 4,
    "anthropic": 5, "claude": 5, "opus": 4,
    "meta": 3, "llama": 5, "llama 3": 6, "llama 3.1": 6,
    "mistral": 4, "mixtral": 4, "qwen": 4, "phi": 3, "sora": 4,
    # Themes
    "world model": 5, "alignment": 3, "ai act": 4, "regulation": 3,
    "fine-tuning": 3, "inference": 3, "open weights": 5, "safety": 3,
    "benchmark": 2, "reasoning": 3, "search": 2, "compute": 3,
    "api": 2, "release": 4, "unveils": 3, "launches": 3, "announces": 3,
}

SOURCE_WEIGHTS = {
    "openai.com": 1.2,
    "deepmind.google": 1.2,
    "ai.googleblog.com": 1.1,
    "theverge.com": 1.05,
    "arstechnica.com": 1.05,
    "technologyreview.com": 1.05,
    "news.mit.edu": 1.05,
}

# -------------------- Data types --------------------

@dataclass
class Item:
    title: str
    link: str
    published: dt.datetime
    source_title: str
    source_domain: str
    summary_html: str = ""
    text: str = ""
    score: float = 0.0


# -------------------- Utilities --------------------

def now_utc() -> dt.datetime:
    return dt.datetime.now(tz=dt.timezone.utc)

def to_utc(dt_struct) -> dt.datetime:
    # feedparser gives time.struct_time; convert to aware UTC dt
    if not dt_struct:
        return now_utc()
    return dt.datetime(*dt_struct[:6], tzinfo=dt.timezone.utc)

def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""

def clean_text(txt: str) -> str:
    txt = unescape(txt or "")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def jaccard_title_sim(a: str, b: str) -> float:
    aset = set(re.findall(r"\w+", a.lower()))
    bset = set(re.findall(r"\w+", b.lower()))
    if not aset or not bset: return 0.0
    return len(aset & bset) / len(aset | bset)

def keyword_score(s: str) -> float:
    s = s.lower()
    score = 0.0
    for kw, w in KEYWORD_WEIGHTS.items():
        if kw in s:
            # weight times rough frequency
            score += w * s.count(kw)
    return score

def recency_score(published: dt.datetime, half_life_hours: float = 48.0) -> float:
    hours = (now_utc() - published).total_seconds() / 3600.0
    return math.exp(-hours / half_life_hours)

def rank_score(item: Item) -> float:
    base = keyword_score(f"{item.title} {item.summary_html}")
    rec = recency_score(item.published)
    domw = SOURCE_WEIGHTS.get(item.source_domain, 1.0)
    return base * 1.0 + rec * 3.0 + (domw - 1.0) * 2.0

# -------------------- Fetch feeds --------------------

def google_news_urls() -> List[str]:
    # Build a handful of Google News RSS search feeds and merge them
    # (q=..., localization via hl/gl/ceid)
    base = "https://news.google.com/rss/search?"
    urls = []
    for q in GOOGLE_NEWS_QUERIES:
        urls.append(f"{base}q={q}&{GOOGLE_NEWS_PARAMS}")
    return urls

def parse_feed(url: str) -> List[Item]:
    out: List[Item] = []
    feed = feedparser.parse(url)
    src_title = feed.feed.get("title", domain_of(url)) or domain_of(url)
    for e in feed.entries:
        link = e.get("link") or e.get("id")
        title = clean_text(e.get("title", ""))
        if not (link and title):
            continue
        # Prefer published_parsed; fall back to updated
        pub = to_utc(e.get("published_parsed") or e.get("updated_parsed"))
        out.append(Item(
            title=title,
            link=link,
            published=pub,
            source_title=src_title,
            source_domain=domain_of(link),
            summary_html=e.get("summary", "") or e.get("description", "") or "",
        ))
    return out

def gather_candidates(days: int, max_candidates: int) -> List[Item]:
    urls = google_news_urls() + CURATED_RSS
    items: List[Item] = []
    cutoff = now_utc() - dt.timedelta(days=days)
    for u in urls:
        try:
            for it in parse_feed(u):
                if it.published >= cutoff:
                    items.append(it)
        except Exception:
            continue
    # Dedup near-identical titles (keep highest-scoring later)
    pruned: List[Item] = []
    for it in items:
        if not any(jaccard_title_sim(it.title, x.title) > 0.8 for x in pruned):
            pruned.append(it)
    # Rank preliminarily on titles/summaries
    for it in pruned:
        it.score = rank_score(it)
    pruned.sort(key=lambda x: x.score, reverse=True)
    return pruned[:max_candidates]

# -------------------- Extract article text --------------------

def extract_text(url: str) -> str:
    # Respectful single shot fetch+extract via trafilatura
    # If blocked or fails, return empty string and we'll fall back to feed summary.
    try:
        dl = trafilatura.fetch_url(url)
        if not dl:
            return ""
        txt = trafilatura.extract(
            dl,
            include_comments=False,
            favor_recall=True,
            output="txt",
        )
        return clean_text(txt or "")
    except Exception:
        return ""

def enrich_with_text(items: List[Item], min_chars=800) -> List[Item]:
    out: List[Item] = []
    seen_domains: Dict[str, int] = defaultdict(int)
    for it in items:
        txt = extract_text(it.link)
        if not txt:
            # fallback to feed summary if extraction fails
            txt = clean_text(re.sub("<[^<]+?>", " ", it.summary_html))  # strip tags
        it.text = txt[:MAX_CHARS_PER_ARTICLE]
        # keep variety: avoid taking more than 2 from same domain at this step
        if len(it.text) >= min_chars and seen_domains[it.source_domain] < 2:
            out.append(it)
            seen_domains[it.source_domain] += 1
    return out

def pick_final_three(candidates: List[Item]) -> List[Item]:
    # Re-rank now that we have bodies: boost substantial content
    for it in candidates:
        body_bonus = min(len(it.text) / 4000.0, 1.0)  # up to +1
        it.score += body_bonus
    candidates.sort(key=lambda x: x.score, reverse=True)

    chosen: List[Item] = []
    for it in candidates:
        if len(chosen) == 3: break
        if any(jaccard_title_sim(it.title, c.title) > 0.7 for c in chosen):
            continue
        # Prefer domain diversity among top picks
        if sum(1 for c in chosen if c.source_domain == it.source_domain) >= 1:
            # allow 1 per domain in the final 3
            continue
        chosen.append(it)

    # If we didn’t reach 3, just fill with next best (even same domain)
    i = 0
    while len(chosen) < 3 and i < len(candidates):
        c = candidates[i]
        if not any(jaccard_title_sim(c.title, x.title) > 0.7 for x in chosen):
            if c not in chosen:
                chosen.append(c)
        i += 1
    return chosen[:3]

# -------------------- OpenAI helpers --------------------

def openai_client() -> OpenAI:
    return OpenAI()  # reads OPENAI_API_KEY from env by default

def call_structured_json(
    client: OpenAI,
    *,
    model: str,
    instructions: str,
    payload: str,
    schema_name: str,
    schema: Dict[str, Any],
    temperature: float = 0.5,
) -> Dict[str, Any]:
    resp = client.responses.create(
        model=model,
        temperature=temperature,
        instructions=instructions,
        input=payload,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema, "strict": True},
        },
    )
    return json.loads(resp.output_text)

# -------------------- Schemas + prompts --------------------

TLDR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["tldr_bullets", "news"],
    "properties": {
        "tldr_bullets": {
            "type": "array",
            "minItems": 3, "maxItems": 5,
            "items": {"type": "string", "maxLength": 180},
        },
        "news": {
            "type": "array",
            "minItems": 3, "maxItems": 3,
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
You will receive exactly three articles with titles, sources, URLs and bodies.

Produce strict JSON with:
- `tldr_bullets`: 3–5 concise bullets (<= 180 chars each) capturing the week.
- `news`: exactly 3 entries; for each:
    - `headline`: short, neutral (<= 100 chars)
    - `summary`: 2–5 factual sentences; no hype.

Rules:
- Use only the provided articles; no external facts.
- Prefer present tense and concrete details.
- No emojis, markdown, or marketing fluff.
- Output *only* JSON conforming to the schema."""

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
Input: the newsletter TL;DR bullets.
Write:
- `subject`: <= {SUBJECT_CHAR_LIMIT} chars; clear, specific, non-clickbait.
- `preheader`: <= {PREHEADER_CHAR_LIMIT} chars; complements the subject.

Rules:
- No ALL CAPS, no emojis, no square brackets.
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

POTW_INSTRUCTIONS = """You design a weekly, practical LLM prompt.
Using the TL;DR + headlines, produce fields:
- `title` (<=120 chars)
- `scenario` (concrete)
- `task` (what to produce)
- `instruction` (constraints & style)

Tone: pragmatic, specific, slightly creative. No emojis. JSON only."""

# -------------------- Rendering --------------------

def render_markdown(date_str: str, picks: List[Item],
                    tldr: Dict[str, Any],
                    subject_block: Dict[str, str],
                    potw: Dict[str, str]) -> str:
    lines: List[str] = []
    lines += ["Header", NEWSLETTER_TITLE, f"Date: {date_str}", ""]
    lines += ["Summary TL;DR"]
    for b in tldr["tldr_bullets"]:
        lines.append(f"- {b}")
    lines += ["", "AI News"]
    for i, item in enumerate(tldr["news"]):
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
        "Sources",
    ]
    for it in picks:
        lines.append(f"- {it.title} — {it.source_title} ({it.link})")
    lines.append("")
    return "\n".join(lines)

# -------------------- Main --------------------

def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description="Automatic AI newsletter generator (3 LLM calls).")
    p.add_argument("--days", type=int, default=7, help="Time window to consider (days)")
    p.add_argument("--max-candidates", type=int, default=40, help="Max items to score from feeds")
    p.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model (default from $NEWSLETTER_MODEL or gpt-4o-mini)")
    p.add_argument("--date", default=dt.date.today().strftime("%B %d, %Y"), help="Human-readable date for header")
    p.add_argument("-o", "--output", type=Path, default=Path("newsletter.md"), help="Output Markdown file")
    p.add_argument("--temp", type=float, default=0.5, help="Generation temperature")
    args = p.parse_args(argv)

    print("Gathering candidates…")
    candidates = gather_candidates(days=args.days, max_candidates=args.max_candidates)
    if not candidates:
        print("No candidates found. Try increasing --days.", file=sys.stderr)
        return 2

    print(f"Scored {len(candidates)} candidates. Extracting article text…")
    enriched = enrich_with_text(candidates, min_chars=600)
    if not enriched:
        print("Extraction failed for all items. Try again later.", file=sys.stderr)
        return 3

    picks = pick_final_three(enriched)
    if len(picks) < 3:
        # best effort: pad from enriched/candidates
        seen = {x.link for x in picks}
        for pool in (enriched, candidates):
            for it in pool:
                if it.link not in seen:
                    picks.append(it); seen.add(it.link)
                    if len(picks) == 3: break
            if len(picks) == 3: break

    # Build model payload (articles)
    articles_blob = []
    for i, it in enumerate(picks, 1):
        body = it.text[:MAX_CHARS_PER_ARTICLE]
        articles_blob.append(f"ARTICLE {i}\nTitle: {it.title}\nSource: {it.source_title}\nURL: {it.link}\n\n{body}\n")
    articles_str = "\n\n".join(articles_blob)

    client = openai_client()

    print("LLM #1: TL;DR + per-article blurbs…")
    tldr_payload = call_structured_json(
        client,
        model=args.model,
        instructions=TLDR_INSTRUCTIONS,
        payload=articles_str,
        schema_name="NewsletterTLDR",
        schema=TLDR_SCHEMA,
        temperature=args.temp,
    )

    tl_bullets_str = "\n".join(f"- {b}" for b in tldr_payload["tldr_bullets"])

    print("LLM #2: Subject + preheader…")
    subject_payload = call_structured_json(
        client,
        model=args.model,
        instructions=SUBJECT_INSTRUCTIONS,
        payload=f"TL;DR bullets:\n{tl_bullets_str}",
        schema_name="SubjectSchema",
        schema=SUBJECT_SCHEMA,
        temperature=0.4 if args.temp >= 0.4 else args.temp,
    )

    print("LLM #3: Prompt of the Week…")
    headlines = "\n".join(f"- {n['headline']}" for n in tldr_payload["news"])
    potw_payload = call_structured_json(
        client,
        model=args.model,
        instructions=POTW_INSTRUCTIONS,
        payload=f"TL;DR:\n{tl_bullets_str}\n\nHeadlines:\n{headlines}",
        schema_name="PromptOfTheWeek",
        schema=POTW_SCHEMA,
        temperature=0.6,
    )

    print(f"Rendering newsletter → {args.output}")
    md = render_markdown(args.date, picks, tldr_payload, subject_payload, potw_payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    print("Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

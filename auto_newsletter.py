#!/usr/bin/env python3
"""
auto_newsletter.py — Fully automatic weekly AI newsletter generator.

What it does:
  - Gathers fresh AI news from Google News RSS queries and curated RSS feeds
  - Ranks + deduplicates items and extracts article text
  - Makes exactly 3 OpenAI Responses API calls:
       1) TL;DR bullets (3–5) + 3 “AI News” blurbs from the chosen articles
       2) Subject + preheader (from the TL;DR)
       3) Prompt of the Week (now: 5–7 ideas that improve prompting, each with template+example)
  - Renders a clean Markdown newsletter and lists sources

Quick start:
  pip install -r requirements.txt
  export OPENAI_API_KEY="sk-..."
  python auto_newsletter.py --output newsletters/$(date +%Y-%m-%d).md
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse, quote

import feedparser           # RSS parsing
import trafilatura          # article fetching + extraction
from trafilatura.settings import use_config
import requests
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

GOOGLE_NEWS_PARAMS = "hl=en-US&gl=US&ceid=US:en"
GOOGLE_NEWS_QUERIES = [
    # Broader queries ensure variety
    '("artificial intelligence" OR "generative AI" OR "large language model")',
    '(GPT OR "ChatGPT" OR "GPT-5" OR "o1" OR "o4" OR "Sora")',
    '(Gemini OR "Google DeepMind" OR "world model" OR "Genie")',
    '(Claude OR Anthropic OR Opus)',
    '(Llama OR Qwen OR Mixtral OR Mistral OR Phi)',
    '("open weights" OR "open-source" OR "checkpoint" OR "Hugging Face")',
    '("AI Act" OR regulation OR policy OR bill OR law OR FTC OR EU)',
    '(NVIDIA OR GPU OR chips OR TPU OR accelerator OR H100 OR B200 OR GB200)',
    '(funding OR acquisition OR partnership OR revenue OR IPO)',
    '(healthcare OR education OR finance OR robotics OR coding OR search OR assistant)',
]

CURATED_RSS = [
    # Working feeds (keep short; we already hit Google News search feeds)
    "https://arstechnica.com/ai/feed/",
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://venturebeat.com/ai/feed/",
    "https://www.technologyreview.com/feed/",
]

# Keyword weights used in ranking (title + summary)
KEYWORD_WEIGHTS = {
    # Companies / labs / families
    "openai": 5, "gpt": 5, "chatgpt": 4, "gpt-5": 7, "o1": 5, "o4": 5, "sora": 4,
    "google": 3, "deepmind": 5, "gemini": 5, "genie": 4,
    "anthropic": 5, "claude": 5, "opus": 4,
    "meta": 3, "llama": 5,
    "mistral": 4, "mixtral": 4, "qwen": 4, "phi": 3,
    # Themes
    "world model": 5, "alignment": 3, "ai act": 4, "regulation": 3,
    "fine-tuning": 3, "inference": 3, "open weights": 5, "open-source": 5,
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

# Categories to enforce diversity
CATEGORY_KEYWORDS = {
    "models_products": {
        "gpt", "chatgpt", "gemini", "claude", "llama", "mixtral", "qwen", "mistral", "phi",
        "model", "weights", "release", "fine-tuning", "checkpoint", "o1", "o4", "gpt-5", "sora"
    },
    "policy_regulation": {
        "ai act", "regulation", "policy", "law", "bill", "compliance", "government",
        "eu", "ftc", "white house", "sec", "nhtsa", "ofcom", "nist", "iso", "uk", "us"
    },
    "applications": {
        "healthcare", "doctor", "medicine", "education", "classroom", "finance", "bank",
        "trading", "customer", "marketing", "robot", "robotics", "search", "assistant",
        "copilot", "game", "gaming", "enterprise", "productivity", "biotech", "space"
    },
    "hardware": {
        "nvidia", "gpu", "chips", "semiconductor", "tpu", "accelerator", "h100", "b200", "gb200",
        "amd", "intel", "arm", "asic"
    },
    "research_science": {
        "paper", "study", "research", "arxiv", "benchmark", "dataset", "agent", "world model",
        "reasoning", "diffusion", "transformer", "architecture", "pretraining", "rlhf"
    },
    "business_finance": {
        "funding", "raises", "acquisition", "acquires", "merger", "partnership", "revenue",
        "earnings", "ipo", "valuation", "round", "seed", "series", "invest", "startup"
    },
    "safety_ethics": {
        "safety", "alignment", "bias", "fairness", "harm", "misinformation", "guardrail",
        "red team", "evals", "privacy", "security", "governance", "responsible ai"
    },
    "open_source": {
        "open-source", "open weights", "github", "mit license", "apache", "model card",
        "hugging face", "checkpoint", "dataset release"
    },
    "tools_platforms": {
        "api", "sdk", "platform", "azure", "aws", "gcp", "vertex", "bedrock", "langchain",
        "vllm", "inference server", "serving", "kubernetes", "k8s"
    },
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
    category: str = "general"

# -------------------- Networking / extraction --------------------

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
})
TRAFI_CFG = use_config()
TRAFI_CFG.set("DEFAULT", "USER_AGENT", SESSION.headers["User-Agent"])

# -------------------- Utilities --------------------

def now_utc() -> dt.datetime:
    return dt.datetime.now(tz=dt.timezone.utc)

def to_utc(dt_struct) -> dt.datetime:
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

def resolve_redirect(url: str) -> str:
    try:
        if "news.google.com" in domain_of(url):
            r = SESSION.get(url, timeout=12, allow_redirects=True)
            return r.url or url
        return url
    except Exception:
        return url

# -------------------- Fetch feeds --------------------

def google_news_urls() -> List[str]:
    base = "https://news.google.com/rss/search?"
    return [f"{base}q={quote(q)}&{GOOGLE_NEWS_PARAMS}" for q in GOOGLE_NEWS_QUERIES]

def parse_feed(url: str) -> List[Item]:
    out: List[Item] = []
    feed = feedparser.parse(url)
    src_title = feed.feed.get("title", domain_of(url)) or domain_of(url)
    for e in feed.entries:
        link = resolve_redirect(e.get("link") or e.get("id") or "")
        title = clean_text(e.get("title", ""))
        if not (link and title): continue
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
    # Dedup near-identical titles
    pruned: List[Item] = []
    for it in items:
        if not any(jaccard_title_sim(it.title, x.title) > 0.8 for x in pruned):
            pruned.append(it)
    # Rank preliminarily
    for it in pruned:
        it.score = rank_score(it)
    pruned.sort(key=lambda x: x.score, reverse=True)
    return pruned[:max_candidates]

# -------------------- Extract article text --------------------

def extract_text(url: str) -> str:
    try:
        dl = trafilatura.fetch_url(url, config=TRAFI_CFG)
        if dl:
            text = trafilatura.extract(dl, config=TRAFI_CFG, include_comments=False, favor_recall=True, output="txt")
            if text: return clean_text(text)
        # Fallback: requests + extract
        r = SESSION.get(url, timeout=12, allow_redirects=True)
        r.raise_for_status()
        text = trafilatura.extract(r.text, url=url, config=TRAFI_CFG, include_comments=False, favor_recall=True, output="txt")
        if text: return clean_text(text)
        # Very rough fallback: join <p> tags
        content = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', r.text, flags=re.DOTALL|re.IGNORECASE)
        paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', content, flags=re.IGNORECASE|re.DOTALL)
        if paragraphs:
            return clean_text(re.sub("<[^<]+?>", " ", " ".join(paragraphs)))
        return ""
    except Exception:
        return ""

def enrich_with_text(items: List[Item], min_chars: int = 300) -> List[Item]:
    out: List[Item] = []
    seen_domains: Dict[str, int] = defaultdict(int)
    target_count = 12  # aim for a decent pool
    for i, it in enumerate(items):
        if seen_domains[it.source_domain] >= 2:
            continue
        print(f"  Extracting {i+1}/{len(items)}: {it.title[:64]}...", end="", flush=True)
        txt = extract_text(it.link)
        if not txt:
            txt = clean_text(re.sub("<[^<]+?>", " ", it.summary_html))
            print(" (using summary)")
        else:
            print(f" ({len(txt)} chars)")
        it.text = txt[:MAX_CHARS_PER_ARTICLE]
        if len(it.text) >= min_chars:
            out.append(it)
            seen_domains[it.source_domain] += 1
            if len(out) >= target_count:
                print(f"\n  Stopping extraction — have {len(out)} candidates")
                break
    return out

# -------------------- Categorization + diverse selection --------------------

def categorize_text(text: str) -> str:
    """Assign a single category based on keyword hits in title/summary/body."""
    s = text.lower()
    hits = Counter()
    for cat, kws in CATEGORY_KEYWORDS.items():
        for kw in kws:
            if kw in s:
                hits[cat] += 1
    if not hits:
        return "general"
    # pick category with the most hits; ties broken by alphabetical order for determinism
    max_hit = max(hits.values())
    best = sorted([c for c, n in hits.items() if n == max_hit])[0]
    return best

def assign_categories(items: List[Item]) -> None:
    for it in items:
        blob = f"{it.title}\n{it.summary_html}\n{it.text[:2000]}"
        it.category = categorize_text(blob)
def pick_final_three(candidates: List[Item]) -> List[Item]:
    # Bonus for substantial content
    for it in candidates:
        it.score += min(len(it.text) / 4000.0, 1.0)

    # Assign categories
    assign_categories(candidates)

    # Sort by score desc overall
    candidates.sort(key=lambda x: x.score, reverse=True)

    # 1) Build the best item per category
    best_by_cat: Dict[str, Item] = {}
    for it in candidates:
        # keep the highest-scoring per category
        if it.category not in best_by_cat:
            best_by_cat[it.category] = it

    # 2) Choose the top 3 categories by (their best item's) score,
    #    but prefer variety across high-signal buckets.
    priority = [
        "models_products", "policy_regulation", "applications",
        "hardware", "research_science", "business_finance",
        "safety_ethics", "open_source", "tools_platforms", "general",
    ]

    # Candidate categories present in this run, ranked by priority then by score
    present = [c for c in priority if c in best_by_cat]
    present.sort(key=lambda c: (priority.index(c), -best_by_cat[c].score))

    chosen: List[Item] = []
    used_domains = set()

    # Stage A: strictly enforce different categories + domains
    for cat in present:
        it = best_by_cat[cat]
        if it.source_domain in used_domains:  # keep domain variety first
            continue
        if any(jaccard_title_sim(it.title, c.title) > 0.7 for c in chosen):
            continue
        chosen.append(it)
        used_domains.add(it.source_domain)
        if len(chosen) == 3:
            return chosen

    # Stage B: allow same domain but still require new categories
    for cat in present:
        if any(x.category == cat for x in chosen):
            continue
        it = best_by_cat[cat]
        if any(jaccard_title_sim(it.title, c.title) > 0.7 for c in chosen):
            continue
        if it not in chosen:
            chosen.append(it)
            if len(chosen) == 3:
                return chosen

    # Stage C: fill remaining from the global list, avoiding near-dup titles
    for it in candidates:
        if any(jaccard_title_sim(it.title, c.title) > 0.7 for c in chosen):
            continue
        if it not in chosen:
            chosen.append(it)
            if len(chosen) == 3:
                break

    # Final sanity: if we *do* have 3+ categories available, ensure all chosen categories are distinct.
    if len({x.category for x in candidates}) >= 3 and len({x.category for x in chosen}) < 3:
        # Replace lowest-scoring duplicate-category item with next best of a missing category
        missing = [c for c in present if c not in {x.category for x in chosen}]
        for miss_cat in missing:
            repl = best_by_cat[miss_cat]
            # find lowest-scoring item that duplicates a category
            dup_idx = None
            lowest_score = float("inf")
            for idx, item in enumerate(chosen):
                if any(item.category == other.category for other in chosen if other is not item):
                    if item.score < lowest_score:
                        dup_idx = idx; lowest_score = item.score
            if dup_idx is not None:
                chosen[dup_idx] = repl
                if len({x.category for x in chosen}) == 3:
                    break

    return chosen[:3]


# -------------------- OpenAI helpers (Responses API + Structured Outputs) --------------------

def openai_client() -> OpenAI:
    return OpenAI()  # reads OPENAI_API_KEY from env

def _require_keys(obj: dict, required: list[str], label: str):
    missing = [k for k in required if k not in obj]
    if missing:
        raise ValueError(f"{label} missing keys: {missing}")

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
    """
    Prefer Responses API + Structured Outputs. If the installed SDK is older
    and doesn't accept response_format, fall back to Chat Completions with
    response_format={'type':'json_object'} and validate required keys.
    """
    # 1) Try modern Responses API
    try:
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
    except TypeError:
        # Older SDK path
        pass

    # 2) Fallback: Chat Completions with json_object (no schema enforcement).
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": instructions + "\nIMPORTANT: Reply with JSON only."},
            {"role": "user", "content": payload},
        ],
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)

    # Minimal validation using 'required' from the provided schema
    if schema_name == "NewsletterTLDR":
        _require_keys(data, ["tldr_bullets", "news"], "TLDR")
        for item in data["news"]:
            _require_keys(item, ["headline", "summary"], "news item")
    elif schema_name == "SubjectSchema":
        _require_keys(data, ["subject", "preheader"], "subject")
    elif schema_name == "PromptOfTheWeek":
        _require_keys(data, ["title", "ideas"], "POTW")
        for idea in data["ideas"]:
            _require_keys(idea, ["name", "why_it_works", "template", "example"], "POTW idea")

    return data

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

# >>> NEW: Prompt-of-the-Week schema = ideas for improving prompting
POTW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "ideas"],
    "properties": {
        "title": {"type": "string", "maxLength": 120},
        "ideas": {
            "type": "array",
            "minItems": 5, "maxItems": 7,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "why_it_works", "template", "example"],
                "properties": {
                    "name": {"type": "string", "maxLength": 80},
                    "why_it_works": {"type": "string", "maxLength": 200},
                    "template": {"type": "string", "maxLength": 400},
                    "example": {"type": "string", "maxLength": 400},
                },
            },
        },
    },
}

POTW_INSTRUCTIONS = """You create a Prompt of the Week that teaches better prompting.
Using the TL;DR + headlines for context (to stay timely), output strict JSON with:
- `title`: a punchy umbrella title (<=120 chars)
- `ideas`: 5–7 items, each with:
   - `name`: the technique (e.g., “Role/Task/Data separation”, “Self‑critique without chain‑of‑thought”)
   - `why_it_works`: 1–2 sentences with mechanism or principle
   - `template`: a reusable prompt pattern users can paste (no emojis)
   - `example`: a concrete filled example for a realistic task

Rules:
- Focus on improving prompting quality: structure, constraints, uncertainty handling, evaluation, schema‑guided outputs, few‑shot contrast, etc.
- Do NOT ask the model to expose chain‑of‑thought; prefer “use a hidden scratchpad, return a concise rationale or final answer.”
- Be practical and specific. Output JSON only."""

# -------------------- Rendering --------------------

def render_markdown(date_str: str, picks: List[Item],
                    tldr: Dict[str, Any],
                    subject_block: Dict[str, str],
                    potw: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines += ["Header", NEWSLETTER_TITLE, f"Date: {date_str}", ""]
    lines += ["Summary TL;DR"]
    for b in tldr["tldr_bullets"]:
        lines.append(f"- {b}")
    lines += ["", "AI News"]
    for item in tldr["news"]:
        lines += [item["headline"], item["summary"], ""]
    lines += ["Prompt of the Week", f"Title: {potw['title']}"]
    for idea in potw["ideas"]:
        lines += [
            f"- {idea['name']}",
            f"  Why: {idea['why_it_works']}",
            f"  Template: {idea['template']}",
            f"  Example: {idea['example']}",
        ]
    lines += [
        "",
        f"(Subject) {subject_block['subject']}",
        f"(Preheader) {subject_block['preheader']}",
        "",
        "Sources",
    ]
    for it in picks:
        lines.append(f"- {it.title} — {it.source_title} ({it.link}) [{it.category}]")
    lines.append("")
    return "\n".join(lines)

# -------------------- Main --------------------

def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description="Automatic AI newsletter generator (3 LLM calls).")
    p.add_argument("--min-chars", type=int, default=250, help="Min chars required to keep an article/body (default 250)")
    p.add_argument("--days", type=int, default=7, help="Time window to consider (days)")
    p.add_argument("--max-candidates", type=int, default=60, help="Max items to score from feeds")
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
    enriched = enrich_with_text(candidates, min_chars=args.min_chars)

    if len(enriched) < 3:
        print(f"Only {len(enriched)} full-text articles found; supplementing with feed summaries.")
        seen_links = {it.link for it in enriched}
        seen_domains = defaultdict(int)
        for it in enriched:
            seen_domains[it.source_domain] += 1
        for it in candidates:
            if it.link in seen_links: continue
            if not it.text:
                txt = clean_text(re.sub("<[^<]+?>", " ", it.summary_html))
                if len(txt) >= 80 and seen_domains[it.source_domain] < 2:
                    it.text = txt
                    enriched.append(it)
                    seen_domains[it.source_domain] += 1
                    if len(enriched) >= 8:
                        break
        if len(enriched) < 3:
            print(f"Warning: Only found {len(enriched)} articles total. Try increasing --days or --max-candidates.", file=sys.stderr)
            if not enriched:
                return 3

    # Pick three with strong diversity
    picks = pick_final_three(enriched)
    if len(picks) < 3:
        # best effort fill
        seen = {x.link for x in picks}
        for pool in (enriched, candidates):
            for it in pool:
                if it.link not in seen and not any(jaccard_title_sim(it.title, x.title) > 0.7 for x in picks):
                    picks.append(it); seen.add(it.link)
                    if len(picks) == 3: break
            if len(picks) == 3: break

    # Build model payload (articles)
    articles_blob = []
    for i, it in enumerate(picks, 1):
        body = it.text[:MAX_CHARS_PER_ARTICLE]
        articles_blob.append(
            f"ARTICLE {i}\n"
            f"Title: {it.title}\n"
            f"Source: {it.source_title}\n"
            f"Category: {it.category}\n"
            f"URL: {it.link}\n\n{body}\n"
        )
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
        temperature=0.55,
    )

    print(f"Rendering newsletter → {args.output}")
    md = render_markdown(args.date, picks, tldr_payload, subject_payload, potw_payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    print("Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

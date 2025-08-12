
# AI Newsletter – automatic, 3-call generator

This repo fetches top AI news (RSS/Google News), extracts article text, and makes exactly three OpenAI calls to produce:

1) TL;DR bullets + 3 “AI News” blurbs
2) Subject + preheader
3) Prompt of the Week

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY
python auto_newsletter.py --output newsletters/$(date +%Y-%m-%d).md
```

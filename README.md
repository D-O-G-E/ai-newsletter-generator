# AI Newsletter – 3-call generator

Generates a weekly AI newsletter from 3 article texts using exactly three LLM calls:

1) TL;DR + per-article news blurbs
2) Subject + preheader
3) Prompt of the Week

## Quickstart

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."   # or use a .env + python-dotenv
python newsletter_maker.py articles/a1.md articles/a2.md articles/a3.md \
  --date "August 12, 2025" --output newsletter.md --model gpt-4o-mini
```

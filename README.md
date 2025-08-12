# AI Newsletter – Automatic Weekly AI Newsletter Generator

An intelligent newsletter generator that automatically curates and summarizes the latest AI news from multiple sources. It fetches articles from Google News RSS queries and curated feeds, extracts content, and uses OpenAI's GPT models to create a professional weekly newsletter with just 3 API calls.

## Features

- 🔍 **Smart News Gathering**: Fetches AI news from Google News RSS (400+ articles) and curated sources
- 📊 **Intelligent Ranking**: Scores articles based on relevance, recency, and source quality
- 📝 **Content Extraction**: Extracts full article text with multiple fallback methods
- 🤖 **AI-Powered Summaries**: Creates TL;DR bullets, news summaries, and a "Prompt of the Week"
- 💰 **Cost-Efficient**: Uses only 3 OpenAI API calls per newsletter (~$0.003)

## What It Generates

1. **TL;DR Section**: 3-5 concise bullets summarizing the week's AI developments
2. **AI News**: 3 detailed news items with headlines and summaries
3. **Prompt of the Week**: A practical LLM prompt based on current trends
4. **Email Metadata**: Subject line and preheader text

## Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd ai-newsletter

# Install dependencies
pip install -r requirements.txt

# Set up your OpenAI API key
cp .env.example .env
# Edit .env and add your OpenAI API key

# Generate a newsletter
python auto_newsletter.py --output newsletters/$(date +%Y-%m-%d).md
```

## Configuration

### Environment Variables (.env file)

```bash
OPENAI_API_KEY=sk-proj-...          # Your OpenAI API key (required)
NEWSLETTER_MODEL=gpt-4o-mini        # Model to use (default: gpt-4o-mini)
MAX_CHARS_PER_ARTICLE=12000         # Max characters per article (default: 12000)
```

### Command Line Options

```bash
python auto_newsletter.py [OPTIONS]

Options:
  --days N              Time window for "fresh" items (default: 7)
  --max-candidates N    How many items to score before narrowing (default: 40)
  --min-chars N         Min chars required to keep an article/body (default: 250)
  --model MODEL         OpenAI model to use (default: from env or gpt-4o-mini)
  --date "DATE"         Human-readable date for header (default: today)
  --output PATH         Output file path (default: newsletter.md)
  --temp FLOAT          Generation temperature (default: 0.5)
```

### Example Commands

```bash
# Standard weekly newsletter
python auto_newsletter.py --days 7 --output newsletters/weekly.md

# Extended bi-weekly newsletter with more candidates
python auto_newsletter.py --days 14 --max-candidates 100 --output newsletters/biweekly.md

# Quick daily digest with lower thresholds
python auto_newsletter.py --days 1 --min-chars 150 --output newsletters/daily.md
```

## Troubleshooting

### "Incorrect API key provided" Error

**Problem**: Getting authentication errors even with a valid API key in `.env`

**Solution**: Check if an old API key is set in your shell environment:
```bash
# Check current environment
echo $OPENAI_API_KEY

# If it shows an old key, unset it
unset OPENAI_API_KEY

# Then run the script again
python auto_newsletter.py
```

### "Only found 2 articles total"

**Problem**: Script finding very few articles despite many candidates

**Solutions**:
1. The script now properly encodes Google News queries and gets 400+ candidates
2. If still having issues:
   - Lower the minimum character threshold: `--min-chars 100`
   - Increase the candidate pool: `--max-candidates 80`
   - The script will automatically use RSS summaries as fallback

### Text Extraction Timeouts

**Problem**: Script times out during article extraction

**Solutions**:
1. The script now stops after extracting 12 good articles
2. Use fewer candidates: `--max-candidates 30`
3. Lower timeout is set (10s) to prevent hanging on slow sites

### Rate Limiting

If you encounter OpenAI rate limits:
1. Use a smaller model: `--model gpt-3.5-turbo`
2. Reduce temperature: `--temp 0.3`
3. Add delays between runs if automating

## How It Works

1. **Gathering**: Queries 6 Google News RSS feeds (~600 items) plus curated sources
2. **Ranking**: Scores articles based on:
   - Keyword relevance (GPT, Claude, Gemini, etc.)
   - Recency (exponential decay with 48-hour half-life)
   - Source quality (trusted domains get a boost)
3. **Extraction**: Attempts to extract full article text using:
   - Trafilatura library (primary method)
   - Direct HTTP fetch with fallbacks
   - RSS summary text (automatic fallback)
4. **Selection**: Picks top 3 articles ensuring diversity
5. **Generation**: Makes 3 OpenAI API calls to create newsletter content

## RSS Sources

### Google News Queries (URL-encoded, ~100 items each)
- "artificial intelligence" OR "generative AI" OR "large language model"
- GPT, ChatGPT, GPT-5, o1, o4
- Gemini, Google DeepMind, world models
- Claude, Anthropic, Opus
- Llama, Qwen, Mixtral, Mistral
- Open weights, AI Act, AI safety, compute

### Curated Feeds
- Ars Technica AI
- TechCrunch AI
- VentureBeat AI
- MIT Technology Review

## Performance

- **Feed gathering**: ~5 seconds for 400+ articles
- **Text extraction**: ~30-60 seconds (stops at 12 good articles)
- **AI generation**: ~10 seconds for all 3 calls
- **Total time**: ~1-2 minutes per newsletter

## Cost Estimation

Using `gpt-4o-mini` (default):
- ~5,000 tokens per newsletter
- Cost: ~$0.003 per newsletter
- Monthly (4 newsletters): ~$0.012

## Development

### Debug Mode

Use the included debug script to troubleshoot extraction issues:
```bash
python debug_newsletter.py
```

### Adding New Sources

Edit `CURATED_RSS` in `auto_newsletter.py`:
```python
CURATED_RSS = [
    "https://example.com/ai/feed.xml",  # Add your feed here
    # ...
]
```

### Customizing Keywords

Modify `KEYWORD_WEIGHTS` to adjust ranking priorities:
```python
KEYWORD_WEIGHTS = {
    "your_keyword": 5,  # Higher weight = higher priority
    # ...
}
```

## Recent Improvements

- Fixed Google News URL encoding to properly fetch 400+ articles
- Added automatic RSS summary fallback when extraction fails
- Optimized extraction to stop after 12 good articles (prevents timeouts)
- Updated curated RSS feeds to remove broken sources
- Improved error handling and progress reporting

## License

This project is provided as-is for educational and personal use.
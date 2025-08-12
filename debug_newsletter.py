#!/usr/bin/env python3
"""Debug version of auto_newsletter.py to troubleshoot extraction failures"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import auto_newsletter
sys.path.insert(0, str(Path(__file__).parent))

from auto_newsletter import (
    gather_candidates, enrich_with_text, extract_text, 
    openai_client, pick_final_three, now_utc
)
import datetime as dt

def test_api_key():
    """Test if OpenAI API key is properly loaded"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return False
    
    if api_key.startswith("sk-"):
        print(f"✅ API key found: {api_key[:7]}...")
        return True
    else:
        print(f"⚠️  API key found but doesn't start with 'sk-': {api_key[:7]}...")
        return True

def test_extraction():
    """Test article extraction with debug output"""
    print("\n=== Testing Article Extraction ===\n")
    
    # First test with a known working URL
    test_urls = [
        "https://www.theverge.com/2024/1/1/test-article",  # May not exist
        "https://openai.com/blog/",  # OpenAI blog main page
        "https://www.anthropic.com/news/claude-3-5-sonnet",  # Known article
    ]
    
    print("Testing extraction on sample URLs:")
    for url in test_urls:
        print(f"\nTesting: {url}")
        try:
            text = extract_text(url)
            if text:
                print(f"✅ Extracted {len(text)} characters")
                print(f"   Preview: {text[:100]}...")
            else:
                print("❌ No text extracted")
        except Exception as e:
            print(f"❌ Error: {type(e).__name__}: {e}")
    
    print("\n=== Testing Feed Gathering ===\n")
    
    # Test gathering candidates
    print("Gathering candidates (7 days, max 10 for debug)...")
    candidates = gather_candidates(days=7, max_candidates=10)
    
    if not candidates:
        print("❌ No candidates found!")
        return
    
    print(f"✅ Found {len(candidates)} candidates\n")
    
    # Show first few candidates
    for i, item in enumerate(candidates[:3]):
        print(f"Candidate {i+1}:")
        print(f"  Title: {item.title}")
        print(f"  URL: {item.link}")
        print(f"  Domain: {item.source_domain}")
        print(f"  Published: {item.published}")
        print(f"  Score: {item.score:.2f}")
        print()
    
    print("\n=== Testing Text Enrichment ===\n")
    
    # Test enrichment with detailed logging
    enriched = []
    failed_extractions = []
    
    for i, item in enumerate(candidates[:5]):  # Test first 5
        print(f"\nExtracting {i+1}/{min(5, len(candidates))}: {item.title[:60]}...")
        print(f"  URL: {item.link}")
        
        try:
            text = extract_text(item.link)
            if text:
                item.text = text
                print(f"  ✅ Extracted {len(text)} chars")
                enriched.append(item)
            else:
                print(f"  ❌ No text extracted")
                failed_extractions.append(item)
        except Exception as e:
            print(f"  ❌ Error: {type(e).__name__}: {e}")
            failed_extractions.append(item)
    
    print(f"\n=== Summary ===")
    print(f"Successfully extracted: {len(enriched)}/{min(5, len(candidates))}")
    print(f"Failed extractions: {len(failed_extractions)}")
    
    if failed_extractions:
        print("\nFailed URLs:")
        for item in failed_extractions:
            print(f"  - {item.link}")

def main():
    print("=== Newsletter Debug Tool ===\n")
    
    # Test API key first
    if not test_api_key():
        print("\n⚠️  Fix the API key issue first!")
        return 1
    
    # Test extraction
    test_extraction()
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
"""
Download Full Movies from Goojara.to using the goojara-wootly-url-extractor tool.

Flow:
  1. Use Puppeteer (via Node.js tool) to search goojara for each movie
  2. Extract the wootly.ch direct download URL
  3. Download using yt-dlp

Prerequisites:
  - Node.js >= 16.x with Chrome installed
  - npm install done in tools/goojara-extractor/
  - pip install yt-dlp

Usage:
  conda activate videorag
  python scripts/download_goojara_movies.py
"""

import os
import sys
import json
import time
import subprocess
import re
from pathlib import Path

try:
    import yt_dlp

    YTDLP_OK = True
except ImportError:
    YTDLP_OK = False

# ============================================================
# CONFIGURATION
# ============================================================
GOOJARA_BASE = "https://ww1.goojara.to"
EXTRACTOR_DIR = Path(r"D:\Study\School\project_ky4\tools\goojara-extractor")
DOWNLOAD_DIR = Path(r"D:\Study\School\project_ky4\data\raw_videos")
PROGRESS_FILE = DOWNLOAD_DIR / "_goojara_progress.json"

# Movies to download (items 7-1 from dvds.txt)
MOVIES = [
    {"imdb_id": "tt0790636", "title": "Dallas Buyers Club", "year": 2013},
    {"imdb_id": "tt0109830", "title": "Forrest Gump", "year": 1994},
    {"imdb_id": "tt0822832", "title": "Marley and Me", "year": 2008},
    {"imdb_id": "tt0073486", "title": "One Flew Over the Cuckoos Nest", "year": 1975},
    {"imdb_id": "tt0110912", "title": "Pulp Fiction", "year": 1994},
    {"imdb_id": "tt0286106", "title": "Signs", "year": 2002},
    {"imdb_id": "tt1045658", "title": "Silver Linings Playbook", "year": 2012},
    {"imdb_id": "tt1385826", "title": "The Adjustment Bureau", "year": 2011},
    {"imdb_id": "tt0106918", "title": "The Firm", "year": 1993},
    {"imdb_id": "tt0068646", "title": "The Godfather", "year": 1972},
    {"imdb_id": "tt1454029", "title": "The Help", "year": 2011},
    {"imdb_id": "tt1189340", "title": "The Lincoln Lawyer", "year": 2011},
    {"imdb_id": "tt0167404", "title": "The Sixth Sense", "year": 1999},
    {"imdb_id": "tt1285016", "title": "The Social Network", "year": 2010},
    {"imdb_id": "tt1142988", "title": "The Ugly Truth", "year": 2009},
    {"imdb_id": "tt0108160", "title": "Sleepless in Seattle", "year": 1993},
    {"imdb_id": "tt0120338", "title": "Titanic", "year": 1997},
    {
        "imdb_id": "tt0241527",
        "title": "Harry Potter and the Sorcerers Stone",
        "year": 2001,
    },
    {
        "imdb_id": "tt0097576",
        "title": "Indiana Jones and the Last Crusade",
        "year": 1989,
    },
    {"imdb_id": "tt0317198", "title": "Bridget Jones The Edge of Reason", "year": 2004},
]


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"completed": [], "failed": [], "wootly_urls": {}, "goojara_urls": {}}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def search_goojara_url(title, year):
    """
    Use Puppeteer to search goojara and find the movie page URL.
    Returns the goojara movie page URL or None.
    """
    search_script = f"""
    const puppeteer = require('puppeteer-core');
    (async () => {{
        const browser = await puppeteer.launch({{ headless: 'new', channel: 'chrome' }});
        try {{
            const page = await browser.newPage();
            await page.goto('{GOOJARA_BASE}', {{ timeout: 60000 }});
            await page.waitForNetworkIdle({{ concurrency: 2 }});
            
            // Find and type into search
            const searchInput = await page.$('input[type="search"], input[name="q"], .form-control, input');
            if (searchInput) {{
                await searchInput.type('{title}');
                await page.keyboard.press('Enter');
                await page.waitForNetworkIdle({{ concurrency: 2, timeout: 15000 }}).catch(() => {{}});
                await new Promise(r => setTimeout(r, 3000));
            }}
            
            // Find movie links  
            const links = await page.$$eval('a[href*="/m"]', els => 
                els.map(e => ({{ href: e.href, text: e.textContent.trim() }}))
            );
            
            // Find best match
            const titleLower = '{title}'.toLowerCase();
            const yearStr = '{year}';
            let bestMatch = null;
            for (const link of links) {{
                const text = link.text.toLowerCase();
                if (text.includes(titleLower.split(' ')[0]) && 
                    (text.includes(yearStr) || !yearStr)) {{
                    bestMatch = link.href;
                    break;
                }}
            }}
            
            if (!bestMatch && links.length > 0) {{
                // Fallback: first movie link
                for (const link of links) {{
                    if (link.href.includes('/m') && !link.href.includes('/movies')) {{
                        bestMatch = link.href;
                        break;
                    }}
                }}
            }}
            
            console.log(JSON.stringify({{ url: bestMatch, results: links.length }}));
        }} catch(e) {{
            console.log(JSON.stringify({{ url: null, error: e.message }}));
        }} finally {{
            await browser.close();
        }}
    }})();
    """

    try:
        result = subprocess.run(
            ["node", "-e", search_script],
            capture_output=True,
            text=True,
            timeout=200,
            cwd=str(EXTRACTOR_DIR),
        )

        output = result.stdout.strip()
        if output:
            # Find the last JSON line
            for line in reversed(output.split("\n")):
                line = line.strip()
                if line.startswith("{"):
                    data = json.loads(line)
                    return data.get("url")
    except Exception as e:
        print(f"  [WARN] Search error: {e}")

    return None


def extract_wootly_url(goojara_url):
    """
    Use the goojara-wootly-url-extractor to get the direct download link.
    """
    try:
        result = subprocess.run(
            ["node", "index.js", "expose", goojara_url, "-v", "debug"],
            capture_output=True,
            text=True,
            timeout=3000,
            cwd=str(EXTRACTOR_DIR),
        )

        output = result.stdout + result.stderr

        # Extract the Link line
        for line in output.split("\n"):
            if "Link:" in line or "LINK:" in line:
                url_match = re.search(r"https?://\S+", line)
                if url_match:
                    return url_match.group(0)

        # Also try to find any wootly/go.wootly URL
        urls = re.findall(r"https?://go\.wootly\.ch/\S+", output)
        if urls:
            return urls[-1]

    except subprocess.TimeoutExpired:
        print(f"  [WARN] Extraction timed out")
    except Exception as e:
        print(f"  [WARN] Extraction error: {e}")

    return None


def download_video(url, output_path):
    """Download video using yt-dlp."""
    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "outtmpl": str(output_path),
        "noplaylist": True,
        "quiet": False,
        "merge_output_format": "mp4",
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return True
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        return False


def main():
    if not YTDLP_OK:
        print("yt-dlp not installed. Run: pip install yt-dlp")
        sys.exit(1)

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    progress = load_progress()

    print("=" * 60)
    print("GOOJARA FULL MOVIE DOWNLOADER (via goojara-extractor)")
    print(f"Target: {len(MOVIES)} movies")
    print(f"Output: {DOWNLOAD_DIR}")
    print(f"Already completed: {len(progress['completed'])}")
    print("=" * 60)

    for i, movie in enumerate(MOVIES, 1):
        mid = movie["imdb_id"]
        title = movie["title"]
        year = movie["year"]
        output_file = DOWNLOAD_DIR / f"{mid}.mp4"

        if mid in progress["completed"]:
            print(f"\n[{i}/{len(MOVIES)}] {title} ({year}) — already done. Skip.")
            continue

        if output_file.exists() and output_file.stat().st_size > 100 * 1024 * 1024:
            print(
                f"\n[{i}/{len(MOVIES)}] {title} — exists ({output_file.stat().st_size // (1024 * 1024)} MB). Skip."
            )
            progress["completed"].append(mid)
            save_progress(progress)
            continue

        print(f"\n[{i}/{len(MOVIES)}] {title} ({year})")

        # Step 1: Get goojara page URL
        goojara_url = progress.get("goojara_urls", {}).get(mid)
        if not goojara_url:
            print(f"  Step 1: Searching goojara for '{title}'...")
            goojara_url = search_goojara_url(title, year)

        if not goojara_url:
            print(f"   Not found on Goojara.")
            progress["failed"].append(mid)
            save_progress(progress)
            continue

        print(f"  Goojara URL: {goojara_url}")
        progress.setdefault("goojara_urls", {})[mid] = goojara_url
        save_progress(progress)

        # Step 2: Extract wootly download URL
        download_url = progress.get("wootly_urls", {}).get(mid)
        if not download_url:
            print(f"  Step 2: Extracting wootly download URL...")
            download_url = extract_wootly_url(goojara_url)

        if not download_url:
            print(f"   Could not extract download URL.")
            progress["failed"].append(mid)
            save_progress(progress)
            continue

        print(f"  Download URL: {download_url[:80]}...")
        progress.setdefault("wootly_urls", {})[mid] = download_url
        save_progress(progress)

        # Step 3: Download
        print(f"  Step 3: Downloading...")
        success = download_video(download_url, output_file)

        if success and output_file.exists():
            size_mb = output_file.stat().st_size // (1024 * 1024)
            print(f"   {title} downloaded ({size_mb} MB)")
            progress["completed"].append(mid)
        else:
            print(f"   Download failed.")
            progress["failed"].append(mid)

        save_progress(progress)
        time.sleep(2)

    print("\n" + "=" * 60)
    print(f"DONE —  {len(progress['completed'])} /  {len(progress['failed'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import yt_dlp
import re
import time


def extract_video_url_from_vsembed(embed_url, output_path):
    """Extract direct video URL from vsembed.ru embed player using Selenium."""

    # Setup headless Chrome
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])

    driver = webdriver.Chrome(options=options)

    try:
        print(f"  [↗] Loading embed player: {embed_url[:60]}...")
        driver.get(embed_url)

        # Wait for video element or iframe to load
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.TAG_NAME, "video"))
        )
        time.sleep(3)  # Wait for JS to inject sources

        # Try to find video source
        video_url = None

        # Method 1: Check <video><source src="...">
        sources = driver.find_elements(By.CSS_SELECTOR, "video source")
        for src in sources:
            url = src.get_attribute("src")
            if url and (url.endswith(".mp4") or ".m3u8" in url):
                video_url = url
                break

        # Method 2: Check video element src attribute
        if not video_url:
            video_elem = driver.find_element(By.TAG_NAME, "video")
            url = video_elem.get_attribute("src")
            if url and (".mp4" in url or ".m3u8" in url):
                video_url = url

        # Method 3: Scan network-like XHR responses via JS injection
        if not video_url:
            script = """
            () => {
                const urls = [];
                performance.getEntriesByType('resource').forEach(r => {
                    if (r.initiatorType === 'xmlhttprequest' || 
                        r.name.includes('.m3u8') || r.name.includes('.mp4')) {
                        urls.push(r.name);
                    }
                });
                return urls;
            }
            """
            candidate_urls = driver.execute_script(script)
            for url in candidate_urls:
                if "vsembed" in url or ".m3u8" in url or ".mp4" in url:
                    video_url = url
                    break

        if not video_url:
            print("  [] Could not extract video URL")
            return False

        print(f"  [] Found video URL: {video_url[:80]}...")

        # Download with yt-dlp
        ydl_opts = {
            "format": "best[ext=mp4]/best",
            "outtmpl": str(output_path),
            "external_downloader": "aria2c" if shutil.which("aria2c") else None,
            "external_downloader_args": ["-x", "16", "-j", "16", "-k", "1M"]
            if shutil.which("aria2c")
            else [],
            "referer": "https://vsembed.ru/",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        return output_path.exists()

    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")
        return False
    finally:
        driver.quit()


extract_video_url_from_vsembed("https://vsembed.ru/embed/movie?imdb=tt0765128", "otp")

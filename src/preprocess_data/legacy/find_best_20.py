import json
from pathlib import Path

# Paths
UNIFIED_JSON = Path(
    r"D:\Study\School\project_ky4\data\unified_dataset\movierag_dataset.json"
)


def get_movie_score(info):
    score = 0

    # 1. Has Script (+50 points)
    if info.get("sources", {}).get("script", False):
        score += 50

    # 2. Has Subtitle (+30 points)
    if info.get("sources", {}).get("subtitle", False):
        score += 30

    # 3. Plot Length (+1 point per 50 chars, max 20)
    plot_len = (
        len(info.get("plot", ""))
        + len(info.get("overview", ""))
        + len(info.get("storyline", ""))
    )
    score += min(20, plot_len // 50)

    # 4. Cast Size (+2 point per cast member, max 20)
    score += min(20, len(info.get("cast", [])) * 2)

    return score


def main():
    if not UNIFIED_JSON.exists():
        print(f"Error: Could not find {UNIFIED_JSON}")
        return

    with open(UNIFIED_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Score each movie
    scored_movies = []
    for imdb_id, info in data.get("movies", {}).items():
        score = get_movie_score(info)
        scored_movies.append((score, imdb_id, info.get("title", "Unknown")))

    # Sort descending by score
    scored_movies.sort(reverse=True, key=lambda x: x[0])

    print("Top 20 Movies by Data Completeness:")
    print("-" * 50)
    top_20 = scored_movies[:20]
    for i, (score, imdb_id, title) in enumerate(top_20, 1):
        print(f"{i:2d}. {imdb_id} - Score {score:3d} - {title}")

    # Output the list of top 20 IDs for easy copy-pasting
    top_20_ids = [m[1] for m in top_20]
    print("\nArray of Top 20 IDs:")
    print(json.dumps(top_20_ids))


if __name__ == "__main__":
    main()

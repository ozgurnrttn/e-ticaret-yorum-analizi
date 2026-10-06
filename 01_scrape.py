"""Play Store'dan Türkçe yorumları çeker -> data/reviews.csv

Kullanım:
    python 01_scrape.py                 # son 90 gün, uygulama başına en çok 8000 yorum
    python 01_scrape.py --days 30 --max 3000
"""
import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from google_play_scraper import Sort, reviews

# Marka -> Play Store paket adı. Başka sektör için sadece burayı değiştir.
APPS = {
    "Trendyol": "trendyol.com",
    "Hepsiburada": "com.pozitron.hepsiburada",
    "n11": "com.dmall.mfandroid",
}

# Kullanıcı adı ve profil fotoğrafı bilerek alınmıyor.
KEEP = ["reviewId", "content", "score", "thumbsUpCount", "at", "reviewCreatedVersion"]


def fetch(app_id: str, cutoff: datetime, max_n: int) -> list[dict]:
    """En yeniden eskiye doğru, cutoff tarihine ya da max_n'e ulaşana kadar çeker."""
    rows, token = [], None
    while len(rows) < max_n:
        batch, token = reviews(
            app_id,
            lang="tr",
            country="tr",
            sort=Sort.NEWEST,
            count=min(200, max_n - len(rows)),
            continuation_token=token,
        )
        if not batch:
            break
        rows.extend(batch)
        print(f"    {len(rows):>6} yorum, en eski: {batch[-1]['at']:%Y-%m-%d}", end="\r")
        if batch[-1]["at"] < cutoff or token is None or token.token is None:
            break
        time.sleep(1)  # sunucuyu yormamak için
    print()
    return [r for r in rows if r["at"] >= cutoff]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90, help="kaç gün geriye gidilsin")
    ap.add_argument("--max", type=int, default=8000, help="uygulama başına üst sınır")
    ap.add_argument("--out", default="data/reviews.csv")
    args = ap.parse_args()

    cutoff = datetime.now() - timedelta(days=args.days)
    frames = []
    for brand, app_id in APPS.items():
        print(f"{brand} ({app_id})")
        rows = fetch(app_id, cutoff, args.max)
        if not rows:
            raise SystemExit(
                f"{brand} için hiç yorum gelmedi. İnternet bağlantısını ve paket adını kontrol et."
            )
        df = pd.DataFrame(rows).reindex(columns=KEEP)
        df.insert(0, "brand", brand)
        frames.append(df)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = pd.concat(frames, ignore_index=True).rename(
        columns={"score": "stars", "at": "date", "reviewCreatedVersion": "app_version"}
    )
    data.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\n{len(data)} yorum -> {out}")
    print(data.groupby("brand")["date"].agg(["count", "min", "max"]).to_string())


if __name__ == "__main__":
    main()

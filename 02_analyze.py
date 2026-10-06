"""Duygu analizi, veri setine uyarlama, yıldız puanıyla doğrulama, konu etiketleme ve grafikler.

Girdi : data/reviews.csv                    (01_scrape.py çıktısı)
Çıktı : data/reviews_scored.csv             (yorum bazında duygu + konular)
        outputs/summary.json                (sitede kullanılacak tüm rakamlar)
        outputs/*.png                       (3 grafik)
        outputs/yuksek_yildiz_olumsuz.csv   (4-5 yıldız verip şikayet yazanlar, göz kontrolü için)

Kullanım:
    python 02_analyze.py            # skorlanmış dosya varsa onu kullanır
    python 02_analyze.py --rescore  # modeli baştan çalıştırır
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

MODEL = "savasy/bert-base-turkish-sentiment-cased"  # ikili: positive / negative
MIN_PER_POINT = 30  # trend grafiğinde bundan az yorumlu dönemler çizilmez

# ---------------------------------------------------------------- metin

_TR = str.maketrans("çğıöşüâîû", "cgiosuaiu")


def fold(text: str) -> str:
    """Küçük harf + Türkçe karakterleri sadeleştir ('Ürün' ve 'urun' aynı olsun)."""
    return text.replace("İ", "i").lower().translate(_TR)


# Konu sözlüğü: sadeleştirilmiş kelime kökleri. Bir yorum birden çok konuya girebilir.
# "uygulama" ve "ürün" gibi her yorumda geçen genel kelimeler bilerek yok.
TOPICS = {
    "Uygulama Hataları": r"acilmi|acilami|donuyo|donma|kasiyo|takil|atiyo|cokuyo|giremi|giris yap|yuklenmi"
                         r"|hata|guncelle|calismi|kapaniyo|beyaz ekran|yavas|sifre|arayuz",
    "Ürün & Satıcı": r"satici|magaza|kalite|sahte|ori?jinal|cakma|dandik|defolu|kusurlu|arizali|kirik|yirtik"
                     r"|bozuk|eksik|yanlis|farkli",
    "Kargo & Teslimat": r"kargo|teslim|kurye|gecik|dagitim|gelmedi|gelmiyor|ulasmadi|ptt",
    "İade & İptal": r"iade|geri odeme|degisim|iptal|param",
    "Müşteri Hizmetleri": r"musteri hizmet|musteri temsil|destek|temsilci|muhatap|cozum|cagri merkez"
                          r"|asistan|ilgilen|cevap ver|yardimci ol",
    "Fiyat & Kampanya": r"fiyat|pahali|ucuz|indirim|kampanya|kupon|firsat|ucret(?!siz)|zam\b",
    "Arama & Ürün Bulma": r"arama|aratiyo|bulami|bulunamadi|alakasiz|filtre|stok|tukendi",
    "Reklam & Bildirim": r"reklam|bildirim|sms|spam",
    "Güven & Dolandırıcılık": r"doland|guven|hirsiz|calin|sahtekar",
    "Ödeme & Üyelik": r"(?<!geri )odeme|kredi kart|banka kart|kartim|karttan|taksit|cuzdan|bakiye|hesab"
                      r"|uyelik|premium|abone|kesinti",
}
_TOPIC_RE = {name: re.compile(rf"\b(?:{pat})") for name, pat in TOPICS.items()}


def tag_topics(df: pd.DataFrame) -> pd.DataFrame:
    folded = df["content"].map(fold)
    for name, rx in _TOPIC_RE.items():
        df[name] = folded.map(lambda t, rx=rx: bool(rx.search(t)))
    return df


# ---------------------------------------------------------------- duygu


def score_sentiment(texts: list[str], batch_size: int = 32) -> list[float]:
    """Her yorum için pozitif olma olasılığını (0-1) döndürür."""
    import torch
    from transformers import pipeline

    clf = pipeline(
        "sentiment-analysis",
        model=MODEL,
        device=0 if torch.cuda.is_available() else -1,
        truncation=True,
        max_length=128,
    )
    p_pos = []
    for i in range(0, len(texts), 1000):
        for o in clf(texts[i : i + 1000], batch_size=batch_size):
            is_pos = o["label"].lower().startswith("pos") or o["label"] == "LABEL_1"
            p_pos.append(o["score"] if is_pos else 1 - o["score"])
        print(f"  {len(p_pos)}/{len(texts)} yorum skorlandı", end="\r")
    print()
    return p_pos


def calibrate(df: pd.DataFrame) -> pd.Series:
    """BERT skorunu bu veri setine uyarlar.

    Hazır model film/ürün yorumlarıyla eğitildiği için uygulama yorumlarında "memnunum" gibi
    ifadeleri bile olumsuz sayabiliyor. Burada yorum metni (karakter n-gram TF-IDF) ve BERT skoru
    birlikte bir lojistik regresyona girer; etiket yıldız puanıdır (1-2 olumsuz, 4-5 olumlu).
    Skorlar çapraz doğrulamadan gelir: hiçbir yorum, kendi yıldızını görmüş bir modelden skor almaz.
    3 yıldızlı yorumlar eğitime girmez, geri kalanların tamamıyla eğitilen modelden skor alır.
    """
    text = df["content"].map(fold).to_numpy()
    p = df["p_positive"].clip(1e-4, 1 - 1e-4).to_numpy()
    bert = (np.log(p / (1 - p)) / 5).reshape(-1, 1)
    stars = df["stars"].to_numpy()
    labelled = np.flatnonzero(stars != 3)
    y = (stars[labelled] >= 4).astype(int)

    def fit_predict(train, test):
        tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)
        x_train = hstack([tfidf.fit_transform(text[train]), csr_matrix(bert[train])]).tocsr()
        x_test = hstack([tfidf.transform(text[test]), csr_matrix(bert[test])]).tocsr()
        model = LogisticRegression(C=3, max_iter=2000, class_weight="balanced")
        return model.fit(x_train, (stars[train] >= 4).astype(int)).predict_proba(x_test)[:, 1]

    out = np.full(len(df), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(labelled, y):
        out[labelled[te]] = fit_predict(labelled[tr], labelled[te])
    three = np.flatnonzero(stars == 3)
    if len(three):
        out[three] = fit_predict(labelled, three)
    return pd.Series(out, index=df.index)


def validate(df: pd.DataFrame, col: str) -> dict:
    """Modeli yıldız puanına karşı ölç: 1-2 yıldız = olumsuz, 4-5 = olumlu, 3 hariç."""
    v = df[df["stars"] != 3]
    y_true = (v["stars"] >= 4).map({True: "pozitif", False: "negatif"})
    labels = ["negatif", "pozitif"]
    y_pred = (v[col] >= 0.5).map({True: "pozitif", False: "negatif"})
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "n": int(len(v)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 3),
        "confusion_matrix": {
            "labels": labels,
            "rows_true_cols_pred": cm.tolist(),
        },
    }


# ---------------------------------------------------------------- grafikler

SURFACE, INK, INK2, MUTED, GRID = "#1a1a19", "#ffffff", "#c3c2b7", "#898781", "#2c2c2a"
# Renk markayı takip eder; sıra değişse de marka aynı renkte kalır.
BRAND_COLORS = {"Trendyol": "#3987e5", "Hepsiburada": "#d95926", "n11": "#199e70"}
_EXTRA = ["#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]


def color_of(brand: str) -> str:
    if brand not in BRAND_COLORS:
        BRAND_COLORS[brand] = _EXTRA[(len(BRAND_COLORS) - 3) % len(_EXTRA)]
    return BRAND_COLORS[brand]


def new_ax(title: str, subtitle: str, size=(9, 5)):
    fig, ax = plt.subplots(figsize=size, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#383835")
    ax.tick_params(colors=MUTED, length=0, labelsize=10)
    fig.text(0.02, 0.96, title, color=INK, fontsize=15, fontweight="bold", va="top")
    fig.text(0.02, 0.895, subtitle, color=INK2, fontsize=10.5, va="top")
    return fig, ax


def save(fig, path: Path, top=0.82):
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def pct_axis(axis):
    axis.set_major_formatter(lambda v, _: f"%{v:.0f}")


def chart_negative_share(neg_share: pd.Series, n: pd.Series, out: Path, window: str):
    s = neg_share.sort_values()
    fig, ax = new_ax("Olumsuz yorum oranı", f"Play Store yorumları, {window}", size=(9, 3.6))
    bars = ax.barh(s.index, s.values, height=0.5, color=[color_of(b) for b in s.index])
    for bar, brand in zip(bars, s.index):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f"%{bar.get_width():.1f}".replace(".", ","), color=INK, fontsize=11, va="center")
    ax.set_yticks(range(len(s)), [f"{b}  (n={n[b]:,})".replace(",", ".") for b in s.index])
    ax.tick_params(axis="y", labelcolor=INK2, labelsize=11)
    ax.set_xlim(0, max(s.max() * 1.15, 10))
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    pct_axis(ax.xaxis)
    save(fig, out, top=0.78)


def chart_topics(topic_share: pd.DataFrame, out: Path, window: str):
    """topic_share: satır = konu, sütun = marka, değer = olumsuz yorumlar içindeki pay (%)."""
    t = topic_share.loc[topic_share.mean(axis=1).sort_values().index]
    brands = list(t.columns)
    fig, ax = new_ax(
        "İnsanlar neden şikayet ediyor?",
        f"Olumsuz yorumların yüzde kaçı bu konudan bahsediyor · {window}",
        size=(9, 1.4 + 0.62 * len(t)),
    )
    h = 0.8 / len(brands)
    for i, brand in enumerate(brands):
        y = [k + (len(brands) - 1 - i) * h - 0.4 + h / 2 for k in range(len(t))]
        ax.barh(y, t[brand].values, height=h, color=color_of(brand),
                edgecolor=SURFACE, linewidth=2, label=brand)
        top = t[brand].idxmax()  # sadece markanın en büyük konusunu etiketle
        k = list(t.index).index(top)
        ax.text(t.loc[top, brand] + 0.6, y[k], f"%{t.loc[top, brand]:.0f}",
                color=INK, fontsize=10, va="center")
    ax.set_yticks(range(len(t)), t.index)
    ax.tick_params(axis="y", labelcolor=INK2, labelsize=11)
    ax.set_xlim(0, t.values.max() * 1.12)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    pct_axis(ax.xaxis)
    ax.legend(loc="lower right", frameon=False, labelcolor=INK2, fontsize=10.5)
    save(fig, out, top=1 - 0.95 / fig.get_figheight())


def chart_trend(trend: pd.DataFrame, period_label: str, out: Path):
    """trend: satır = dönem başlangıcı, sütun = marka, değer = olumsuz oran (%)."""
    fig, ax = new_ax(
        f"Olumsuz yorum oranı, {period_label}",
        f"En az {MIN_PER_POINT} yorumu olan dönemler",
    )
    for brand in trend.columns:
        s = trend[brand].dropna()
        if s.empty:
            continue
        ax.plot(s.index, s.values, color=color_of(brand), linewidth=2, label=brand,
                marker="o" if len(s) < 3 else None, markersize=6)
        ax.annotate(f"%{s.iloc[-1]:.0f}", (s.index[-1], s.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", color=INK, fontsize=10, va="center")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, None)
    ax.margins(x=0.06)
    pct_axis(ax.yaxis)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d.%m"))
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.02), ncols=len(trend.columns),
              frameon=False, labelcolor=INK2, fontsize=10.5)
    save(fig, out, top=0.86)


# ---------------------------------------------------------------- akış


def load_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["content"] = df["content"].fillna("").astype(str).str.strip()
    df = df[df["content"].str.contains(r"\w", regex=True)]
    df = df.drop_duplicates(subset=["brand", "content", "date"])
    # Tüm markalar için ortak dönem: en geç başlayan markanın ilk gününden itibaren.
    start = df.groupby("brand")["date"].min().max().normalize()
    return df[df["date"] >= start].reset_index(drop=True)


def main(score_fn=score_sentiment) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/reviews.csv")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--rescore", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scored_path = Path(args.data).with_name("reviews_scored.csv")

    df = load_clean(Path(args.data))
    cached = None if args.rescore or not scored_path.exists() else pd.read_csv(scored_path)
    if cached is not None and set(df["reviewId"]) <= set(cached["reviewId"]):
        df = df.merge(cached[["reviewId", "p_positive"]].drop_duplicates("reviewId"), on="reviewId")
        print(f"Kayıtlı skorlar kullanıldı ({scored_path})")
    else:
        print(f"{len(df)} yorum skorlanıyor ({MODEL})")
        df["p_positive"] = score_fn(df["content"].tolist())

    df["p_calibrated"] = calibrate(df)
    df["sentiment"] = (df["p_calibrated"] >= 0.5).map({True: "pozitif", False: "negatif"})
    df = tag_topics(df)
    df.to_csv(scored_path, index=False, encoding="utf-8-sig")

    topics = list(TOPICS)
    neg = df[df["sentiment"] == "negatif"]
    long_neg = neg[neg["content"].str.split().str.len() > 3]
    window = f"{df['date'].min():%d.%m.%Y} – {df['date'].max():%d.%m.%Y}"

    n = df.groupby("brand").size()
    neg_share = (df["sentiment"] == "negatif").groupby(df["brand"]).mean() * 100
    topic_share = (neg.groupby("brand")[topics].mean() * 100).T  # konu x marka
    topic_neg_rate = (  # o konudan bahseden yorumların yüzde kaçı olumsuz
        pd.DataFrame({t: (df[df[t]]["sentiment"] == "negatif").groupby(df[df[t]]["brand"]).mean()
                      for t in topics}) * 100
    ).T

    # Yıldız yüksek ama metin şikayet: yıldız ortalamasının göstermediği memnuniyetsizlik.
    high = df[df["stars"] >= 4]
    hidden = high[high["sentiment"] == "negatif"]
    hidden_share = hidden.groupby("brand").size().reindex(n.index, fill_value=0) / high.groupby("brand").size() * 100
    hidden.sort_values("p_calibrated")[["brand", "stars", "date", "content"]].to_csv(
        out / "yuksek_yildiz_olumsuz.csv", index=False, encoding="utf-8-sig")

    span_days = (df["date"].max() - df["date"].min()).days
    weekly = span_days >= 56
    period = df["date"].dt.to_period("W" if weekly else "D").dt.start_time
    grp = (df["sentiment"] == "negatif").groupby([period, df["brand"]])
    trend = (grp.mean() * 100).where(grp.size() >= MIN_PER_POINT).unstack("brand")
    if weekly:  # baştaki ve sondaki yarım haftalar yanıltır, at
        first, last = df["date"].min().normalize(), df["date"].max().normalize()
        trend = trend[(trend.index >= first) & (trend.index + pd.Timedelta(days=6) <= last)]

    chart_negative_share(neg_share, n, out / "1_olumsuz_oran.png", window)
    chart_topics(topic_share, out / "2_sikayet_konulari.png", window)
    chart_trend(trend, "haftalık" if weekly else "günlük", out / "3_trend.png")

    summary = {
        "donem": window,
        "model": MODEL,
        "yorum_sayisi": {b: int(v) for b, v in n.items()},
        "ortalama_yildiz": df.groupby("brand")["stars"].mean().round(2).to_dict(),
        "olumsuz_oran_yuzde": neg_share.round(1).to_dict(),
        "dogrulama_yildiza_karsi": {
            "hazir_model": validate(df, "p_positive"),
            "uyarlanmis_model": validate(df, "p_calibrated"),
        },
        "yuksek_yildizli_olumsuz_yuzde": hidden_share.round(1).to_dict(),
        "olumsuzlarda_konu_payi_yuzde": topic_share.round(1).to_dict(),
        "konuda_olumsuz_oran_yuzde": topic_neg_rate.round(1).to_dict(),
        "en_cok_sikayet_edilen_konu": topic_share.idxmax().to_dict(),
        # Sözlüğün yakalayamadığı olumsuz yorumların payı: yüksekse sözlüğü genişlet.
        "konusu_bulunamayan_olumsuz_yuzde": round(float((~neg[topics].any(axis=1)).mean() * 100), 1),
        # Aynı oran, "berbat" gibi konu taşıyamayacak kadar kısa yorumlar (3 kelime ve altı) hariç.
        "konusu_bulunamayan_olumsuz_yuzde_4_kelime_ustu": round(float((~long_neg[topics].any(axis=1)).mean() * 100), 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()

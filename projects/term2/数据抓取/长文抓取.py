'''【001】新闻采集：按时间窗口获取和筛选长文新闻，输出平衡样本；凭据由本地环境提供。'''
import json
import os
import random
import re
import time
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

try:
    import tushare as ts
except ImportError:
    ts = None


# =========================
# 1) 初始化 Tushare
# =========================
TOKEN = os.getenv("TUSHARE_TOKEN", "").strip()
CUSTOM_URL = os.getenv("TUSHARE_NEWS_API_URL", "").strip()


# =========================
# 2) 参数设置
# =========================
SRC = "新浪财经"
end_dt = datetime.now()
try:
    start_dt = end_dt.replace(year=end_dt.year - 2)
except ValueError:
    start_dt = end_dt.replace(year=end_dt.year - 2, month=2, day=28)

COARSE_DAYS = 30
MIN_SPLIT_HOURS = 6
SLEEP_SECONDS = 0.2
BALANCED_TARGET_TOTAL = 3000
BALANCED_SEED = 42
FAILED_WINDOWS_OUTPUT = "sina_major_news_last_2y_fetch_failures.json"
MONTH_COMPANY_LIMIT = 2
MONTH_COMPANY_OVERFLOW_LIMIT = 3


# =========================
# 3) 关键词设置
# =========================
EVENT_KEYWORDS = {
    "credit_risk": {
        "negative_events": ["债务违约", "违约", "债券违约", "展期失败", "贷款逾期", "流动性危机", "评级下调", "破产重整"],
        "positive_events": ["债务按期兑付", "展期成功", "流动性改善", "债务重组完成"]
    },
    "fraud_risk": {
        "negative_events": ["财务造假", "虚增收入", "虚假披露", "资金占用", "商业贿赂", "利益输送"],
        "positive_events": ["主动披露", "资金占用清偿完成", "审计整改认可"]
    },
    "operational_risk": {
        "negative_events": ["重大安全事故", "停工停产", "系统故障", "数据泄露", "监管处罚"],
        "positive_events": ["复工复产", "恢复运营", "安全整改完成", "合规验收"]
    },
    "market_risk": {
        "negative_events": ["股价暴跌", "闪崩", "利率飙升", "汇率贬值", "行业景气下行"],
        "positive_events": ["股价企稳", "利率波动收敛", "汇率趋稳"]
    }
}

STRONG_RISK_KEYWORDS = sorted(set(kw for r in EVENT_KEYWORDS.values() for g in r.values() for kw in g))
EARLY_WARNING_KEYWORDS = sorted(
    {
        "流动性紧张",
        "偿债压力",
        "兑付压力",
        "融资受阻",
        "资金链承压",
        "现金流恶化",
        "债务展期",
        "商誉减值",
        "大额计提",
        "业绩预亏",
        "订单下滑",
        "客户流失",
        "诉讼增加",
        "应收账款攀升",
        "存货积压",
        "需求疲软",
        "经营承压",
        "亏损扩大",
    }
)
BUSINESS_CHANGE_KEYWORDS = sorted(
    {
        "裁员",
        "关店",
        "缩产",
        "停产检修",
        "复工复产",
        "战略收缩",
        "业务调整",
        "资产出售",
        "引战投",
        "再融资",
        "债务重组",
        "高管变动",
        "控股股东变更",
        "主营波动",
        "毛利率下滑",
        "产能调整",
        "出售股权",
        "引入投资者",
    }
)
KEYWORDS = sorted(set(STRONG_RISK_KEYWORDS + EARLY_WARNING_KEYWORDS + BUSINESS_CHANGE_KEYWORDS))
EXCLUDE_KEYWORDS = ["娱乐", "体育", "电影", "明星", "综艺", "球员", "票房"]
COMPANY_SUFFIXES = ("股份有限公司", "有限责任公司", "有限公司", "股份", "集团", "公司")
COMPANY_PATTERN = re.compile(r"([A-Za-z0-9\u4e00-\u9fa5]{1,30}?(?:股份有限公司|有限责任公司|有限公司|股份|集团|公司))")


# =========================
# 4) 核心处理函数（包含清洗逻辑）
# =========================
def clean_html_content(raw_html):
    """
    针对新浪财经数据的深度清洗：
    1. 去除 HTML 标签、脚本、样式
    2. 移除新浪特定的广告 JS 占位符
    3. 合并多余换行
    """
    if not isinstance(raw_html, str) or not raw_html.strip():
        return ""

    soup = BeautifulSoup(raw_html, "html.parser")
    for element in soup(["script", "style", "video", "iframe", "canvas", "figure"]):
        element.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"\(sinaads\s*=\s*window\.sinaads\s*\|\|\s*\[\]\)\.push\(\{\}\);", "", text)
    text = re.sub(r"article_adlist.*?;", "", text)
    text = re.sub(r"font-L|strong-Bold|color0|align-Center", "", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def looks_like_html(text: Any) -> bool:
    raw = str(text or "")
    lowered = raw.lower()
    return (
        "<" in raw
        and ">" in raw
        and any(marker in lowered for marker in ("<p", "<div", "<span", "<br", "<script", "<style", "<html", "<body"))
    )


def clean_title_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if not looks_like_html(text):
        return text
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MarkupResemblesLocatorWarning)
        return BeautifulSoup(text, "html.parser").get_text().strip()


def fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def build_tushare_client():
    if ts is None:
        raise ModuleNotFoundError("缺少 tushare，请先执行 `python -m pip install tushare`。")
    if not TOKEN:
        raise RuntimeError("请在环境变量中设置 TUSHARE_TOKEN。")
    if not CUSTOM_URL:
        raise RuntimeError("请在环境变量中设置已授权的 TUSHARE_NEWS_API_URL。")
    pro = ts.pro_api(TOKEN)
    pro._DataApi__token = TOKEN
    pro._DataApi__http_url = CUSTOM_URL
    return pro


def safe_major_news(pro, start_time: datetime, end_time: datetime) -> pd.DataFrame:
    errors = []
    field_options = ["title,content,pub_time,src", "title,content"]
    for fields in field_options:
        try:
            df = pro.major_news(src=SRC, start_date=fmt(start_time), end_date=fmt(end_time), fields=fields)
            time.sleep(SLEEP_SECONDS)
            return pd.DataFrame(df) if df is not None else pd.DataFrame()
        except Exception as exc:
            errors.append(f"fields={fields}: {exc}")
    raise RuntimeError(" | ".join(errors))


def fetch_adaptive(pro, start_time: datetime, end_time: datetime, failed_windows: list[Dict[str, str]] | None = None) -> pd.DataFrame:
    try:
        df = safe_major_news(pro, start_time, end_time)
    except Exception as exc:
        if (end_time - start_time) > timedelta(hours=MIN_SPLIT_HOURS):
            mid_time = start_time + (end_time - start_time) / 2
            print(f"[拆分重试] {fmt(start_time)} ~ {fmt(end_time)} | reason={exc}", flush=True)
            return pd.concat(
                [
                    fetch_adaptive(pro, start_time, mid_time, failed_windows=failed_windows),
                    fetch_adaptive(pro, mid_time, end_time, failed_windows=failed_windows),
                ],
                ignore_index=True,
            )
        print(f"[跳过窗口] {fmt(start_time)} ~ {fmt(end_time)} | reason={exc}", flush=True)
        if failed_windows is not None:
            failed_windows.append(
                {
                    "start_time": fmt(start_time),
                    "end_time": fmt(end_time),
                    "reason": str(exc),
                }
            )
        return pd.DataFrame()
    n = len(df)
    print(f"[抓取] {fmt(start_time)} ~ {fmt(end_time)} -> {n} 条")
    if n >= 400 and (end_time - start_time) > timedelta(hours=MIN_SPLIT_HOURS):
        mid_time = start_time + (end_time - start_time) / 2
        return pd.concat(
            [
                fetch_adaptive(pro, start_time, mid_time, failed_windows=failed_windows),
                fetch_adaptive(pro, mid_time, end_time, failed_windows=failed_windows),
            ],
            ignore_index=True,
        )
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "src" not in df.columns and "src_site" in df.columns:
        df = df.rename(columns={"src_site": "src"})
    for col in ["title", "content", "pub_time", "src"]:
        if col not in df.columns:
            df[col] = None
    return df[["title", "content", "pub_time", "src"]].copy()


def keyword_filter(df: pd.DataFrame, include_keywords=None, exclude_keywords=None) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    include_keywords = include_keywords or []
    exclude_keywords = exclude_keywords or []

    out = df.copy()
    text = out["title"].fillna("") + "\n" + out["content"].fillna("")

    if include_keywords:
        include_pattern = "|".join(re.escape(k) for k in include_keywords)
        include_mask = text.str.contains(include_pattern, case=False, regex=True)
    else:
        include_mask = pd.Series([True] * len(out), index=out.index)

    if exclude_keywords:
        exclude_pattern = "|".join(re.escape(k) for k in exclude_keywords)
        exclude_mask = text.str.contains(exclude_pattern, case=False, regex=True)
    else:
        exclude_mask = pd.Series([False] * len(out), index=out.index)

    return out[include_mask & (~exclude_mask)].copy()


def normalize_company_name(name: Any) -> str:
    text = str(name or "").strip()
    text = re.sub(r"[\s\u3000]+", "", text)
    changed = True
    while changed:
        changed = False
        for suffix in COMPANY_SUFFIXES:
            if text.endswith(suffix) and len(text) > len(suffix):
                text = text[: -len(suffix)]
                changed = True
                break
    return text


def extract_company_key(title: Any, content: Any) -> str:
    for text in (str(title or ""), str(content or "")[:1200]):
        match = COMPANY_PATTERN.search(text)
        if match:
            company_name = normalize_company_name(match.group(1))
            if company_name:
                return company_name
    title_text = str(title or "").strip()
    if title_text:
        return re.sub(r"[\s\u3000]+", "", title_text[:16]).lower()
    return ""


def count_keyword_hits(text: str, keywords: List[str]) -> int:
    return sum(text.count(keyword) for keyword in keywords)


def compute_article_priority(row: Dict[str, Any]) -> float:
    title = str(row.get("title", "") or "")
    content = str(row.get("content", "") or "")
    title_text = title.lower()
    full_text = f"{title}\n{content}".lower()

    strong_hits = count_keyword_hits(full_text, STRONG_RISK_KEYWORDS)
    early_hits = count_keyword_hits(full_text, EARLY_WARNING_KEYWORDS)
    change_hits = count_keyword_hits(full_text, BUSINESS_CHANGE_KEYWORDS)
    title_bonus = (
        2 * count_keyword_hits(title_text, STRONG_RISK_KEYWORDS)
        + 1 * count_keyword_hits(title_text, EARLY_WARNING_KEYWORDS)
        + 1 * count_keyword_hits(title_text, BUSINESS_CHANGE_KEYWORDS)
    )
    content_quality = min(len(content) / 800.0, 1.0)
    return 3.0 * strong_hits + 2.0 * early_hits + 1.0 * change_hits + 2.0 * title_bonus + content_quality


def allocate_monthly_targets(month_counts: Dict[str, int], target_total: int) -> Dict[str, int]:
    months = sorted(month_counts.keys())
    if not months or target_total <= 0:
        return {}

    capped_total = min(target_total, sum(month_counts.values()))
    base = capped_total // len(months)
    targets = {month: min(month_counts[month], base) for month in months}
    assigned = sum(targets.values())
    remaining = capped_total - assigned

    while remaining > 0:
        candidates = [month for month in months if month_counts[month] > targets[month]]
        if not candidates:
            break
        candidates.sort(key=lambda month: (month_counts[month] - targets[month], month), reverse=True)
        for month in candidates:
            if remaining <= 0:
                break
            if month_counts[month] <= targets[month]:
                continue
            targets[month] += 1
            remaining -= 1

    return targets


def sample_balanced_dataframe(df: pd.DataFrame, target_total: int, seed: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if df.empty:
        return df.copy(), {
            "input_rows_after_time_filter": 0,
            "target_total": target_total,
            "output_rows": 0,
            "month_count": 0,
            "input_month_counts": {},
            "output_month_counts": {},
        }

    working = df.copy()
    parsed_times = pd.to_datetime(working["pub_time"], errors="coerce")
    working = working.loc[parsed_times.notna()].copy()
    if working.empty:
        return working, {
            "input_rows_after_time_filter": 0,
            "target_total": target_total,
            "output_rows": 0,
            "month_count": 0,
            "input_month_counts": {},
            "output_month_counts": {},
        }

    working["pub_time"] = parsed_times.loc[working.index].dt.strftime("%Y-%m-%d %H:%M:%S")
    working["month"] = parsed_times.loc[working.index].dt.strftime("%Y-%m")

    working["company_key"] = [
        extract_company_key(title=row["title"], content=row["content"])
        for _, row in working.iterrows()
    ]
    working["priority_score"] = [
        compute_article_priority(row)
        for row in working.to_dict("records")
    ]
    input_month_counts = working["month"].value_counts().sort_index().to_dict()
    monthly_targets = allocate_monthly_targets(input_month_counts, target_total)

    sampled_parts = []
    prev_month_companies: set[str] = set()
    for month in sorted(monthly_targets.keys()):
        month_df = working[working["month"] == month].copy()
        keep = monthly_targets[month]
        month_df = month_df.sort_values(
            by=["priority_score", "pub_time", "title"],
            ascending=[False, False, True],
            kind="stable",
        ).reset_index(drop=True)
        if len(month_df) <= keep:
            sampled_parts.append(month_df)
            prev_month_companies = {
                str(item).strip() for item in month_df["company_key"].tolist() if str(item).strip()
            }
            continue

        selected_indices: List[int] = []
        selected_company_counts: Dict[str, int] = defaultdict(int)

        # 第一轮：公司覆盖优先，每个公司先保留 1 篇
        for idx, row in month_df.iterrows():
            company_key = str(row.get("company_key", "")).strip() or f"__row__{idx}"
            if selected_company_counts[company_key] >= 1:
                continue
            selected_indices.append(idx)
            selected_company_counts[company_key] += 1
            if len(selected_indices) >= keep:
                break

        # 第二轮：优先补前一月已经出现过的公司，提升跨月连续性
        if len(selected_indices) < keep and prev_month_companies:
            for idx, row in month_df.iterrows():
                if idx in selected_indices:
                    continue
                company_key = str(row.get("company_key", "")).strip()
                if not company_key or company_key not in prev_month_companies:
                    continue
                if selected_company_counts[company_key] >= MONTH_COMPANY_LIMIT:
                    continue
                selected_indices.append(idx)
                selected_company_counts[company_key] += 1
                if len(selected_indices) >= keep:
                    break

        # 第三轮：按相关度补齐，单公司默认最多 2 篇，必要时可放宽到 3 篇
        for max_per_company in (MONTH_COMPANY_LIMIT, MONTH_COMPANY_OVERFLOW_LIMIT):
            if len(selected_indices) >= keep:
                break
            for idx, row in month_df.iterrows():
                if idx in selected_indices:
                    continue
                company_key = str(row.get("company_key", "")).strip() or f"__row__{idx}"
                if selected_company_counts[company_key] >= max_per_company:
                    continue
                selected_indices.append(idx)
                selected_company_counts[company_key] += 1
                if len(selected_indices) >= keep:
                    break

        sampled_parts.append(month_df.loc[selected_indices].copy())
        prev_month_companies = {
            str(item).strip() for item in month_df.loc[selected_indices, "company_key"].tolist() if str(item).strip()
        }

    sampled_df = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else working.iloc[0:0].copy()
    sampled_df = sampled_df.sort_values(by=["pub_time", "title"], kind="stable").reset_index(drop=True)
    output_month_counts = sampled_df["month"].value_counts().sort_index().to_dict()
    output_unique_company_count = int(sum(1 for item in sampled_df["company_key"].dropna().unique() if str(item).strip()))
    output_company_month_count = int(
        sampled_df[["month", "company_key"]]
        .dropna()
        .assign(company_key=lambda frame: frame["company_key"].astype(str).str.strip())
        .query("company_key != ''")
        .drop_duplicates()
        .shape[0]
    )
    sampled_df = sampled_df[["title", "content", "pub_time", "src"]].copy()

    summary = {
        "input_rows_after_time_filter": len(working),
        "target_total": target_total,
        "output_rows": len(sampled_df),
        "month_count": len(input_month_counts),
        "input_month_counts": input_month_counts,
        "output_month_counts": output_month_counts,
        "output_unique_company_count": output_unique_company_count,
        "output_company_month_count": output_company_month_count,
        "selection_strategy": {
            "monthly_targeting": True,
            "company_coverage_priority": True,
            "continuity_priority": True,
            "per_company_limit": MONTH_COMPANY_LIMIT,
            "overflow_limit": MONTH_COMPANY_OVERFLOW_LIMIT,
        },
    }
    return sampled_df, summary


def clean_fetched_chunk(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    cleaned = normalize_columns(df)
    cleaned = cleaned.drop_duplicates(subset=["title", "pub_time", "src"]).reset_index(drop=True)
    cleaned["content"] = cleaned["content"].apply(clean_html_content)
    cleaned["title"] = cleaned["title"].apply(clean_title_text)
    return cleaned


def row_identity(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("title", "") or "").strip(),
        str(row.get("pub_time", "") or "").strip(),
        str(row.get("src", "") or "").strip(),
    )


def append_unique_rows(
    *,
    target_rows: list[Dict[str, Any]],
    source_df: pd.DataFrame,
    seen_keys: set[Tuple[str, str, str]],
    limit: int | None = None,
) -> None:
    for row in source_df.to_dict("records"):
        key = row_identity(row)
        if key in seen_keys:
            continue
        if limit is not None and len(target_rows) >= limit:
            break
        seen_keys.add(key)
        target_rows.append(row)


def day_iterator(start: datetime, end: datetime):
    cursor = datetime(start.year, start.month, start.day)
    while cursor < end:
        next_day = min(cursor + timedelta(days=1), end)
        yield cursor, next_day
        cursor = next_day


def fetch_day_candidates(
    pro,
    day_start: datetime,
    day_end: datetime,
    failed_windows: list[Dict[str, str]] | None = None,
) -> pd.DataFrame:
    day_filtered_rows: list[Dict[str, Any]] = []
    seen_filtered_keys: set[Tuple[str, str, str]] = set()

    part_df = fetch_adaptive(pro, day_start, day_end, failed_windows=failed_windows)
    cleaned_part = clean_fetched_chunk(part_df)

    filtered_part = keyword_filter(cleaned_part, include_keywords=KEYWORDS, exclude_keywords=EXCLUDE_KEYWORDS)
    append_unique_rows(
        target_rows=day_filtered_rows,
        source_df=filtered_part,
        seen_keys=seen_filtered_keys,
    )

    filtered_df = pd.DataFrame(day_filtered_rows, columns=["title", "content", "pub_time", "src"])
    day_label = day_start.strftime("%Y-%m-%d")
    print(f"[日期完成] {day_label} filtered_kept={len(filtered_df)}", flush=True)
    return filtered_df


def main() -> None:
    pro = build_tushare_client()

    all_filtered_parts = []
    failed_windows: list[Dict[str, str]] = []
    for day_start, day_end in day_iterator(start_dt, end_dt):
        day_filtered_df = fetch_day_candidates(
            pro,
            day_start=day_start,
            day_end=day_end,
            failed_windows=failed_windows,
        )
        all_filtered_parts.append(day_filtered_df)

    if all_filtered_parts:
        filtered_df = pd.concat(all_filtered_parts, ignore_index=True)
    else:
        filtered_df = pd.DataFrame(columns=["title", "content", "pub_time", "src"])
    filtered_df = filtered_df.drop_duplicates(subset=["title", "pub_time", "src"]).reset_index(drop=True)

    balanced_df, balanced_summary = sample_balanced_dataframe(
        filtered_df,
        target_total=BALANCED_TARGET_TOTAL,
        seed=BALANCED_SEED,
    )
    balanced_output = "sina_major_news_last_2y_balanced_3k.csv"
    balanced_summary_output = "sina_major_news_last_2y_balanced_3k.summary.json"
    balanced_df.to_csv(balanced_output, index=False, encoding="utf-8-sig")
    with open(balanced_summary_output, "w", encoding="utf-8") as f:
        json.dump(balanced_summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with open(FAILED_WINDOWS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump({"failed_windows": failed_windows, "failed_count": len(failed_windows)}, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"\n关键词筛选后候选新闻数：{len(filtered_df)}")
    print(f"平衡采样新闻已保存：{balanced_output}，共 {len(balanced_df)} 条")
    print(f"平衡采样摘要已保存：{balanced_summary_output}")
    print(f"失败窗口清单已保存：{FAILED_WINDOWS_OUTPUT}，共 {len(failed_windows)} 个")
    print("\n示例预览：")
    print(balanced_df[["title", "content"]].head(5))


if __name__ == "__main__":
    main()

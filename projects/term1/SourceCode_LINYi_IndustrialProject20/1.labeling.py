import json
import argparse
from typing import List, Dict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from time import sleep

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from llm_config import BAILIAN_BASE_URL, BAILIAN_MODEL, get_bailian_api_key

class RiskType(Enum):
    CREDIT_RISK = "credit_risk"
    MARKET_RISK = "market_risk"
    OPERATIONAL_RISK = "operational_risk"
    FRAUD_RISK = "fraud_risk"

ALLOWED_RISKS = {r.value for r in RiskType}

@dataclass
class RiskAssessment:
    risk_types: List[str]
    confidence_scores: Dict[str, float]
    reasoning: str
    key_indicators: List[str]
    status: str = "ok"
    error_message: str = ""

class AIBasedRiskLabeler:
    def __init__(
        self,
        api_key: str,
        model_name: str = BAILIAN_MODEL,
        base_url: str = BAILIAN_BASE_URL,
        temperature: float = 0.1,
        timeout: int = 60,
    ):
        self.llm = ChatOpenAI(
            temperature=temperature,
            model=model_name,
            timeout=timeout,
            api_key=api_key,
            base_url=base_url,
            extra_body={"enable_thinking": False},
        )

        self.risk_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial risk analyst expert. Analyze news articles and identify potential financial and non-financial risks.

Available risk types (return ONLY these exact keys when applicable):
- credit_risk: Issues related to credit, defaults, bankruptcies, debt restructuring, loan non-payments, credit rating downgrades
- market_risk: Market volatility, price fluctuations, interest rate changes, currency risks, liquidity issues, market crashes
- operational_risk: System failures, IT outages, supply chain disruptions, operational errors, cyber attacks, natural disasters
- fraud_risk: Financial fraud, money laundering, bribery, insider trading, accounting irregularities, sanctions violations

Return a single JSON object with keys:
- "risks": string[]  (subset of the allowed risk types)
- "confidence": object  (key=risk_type, value=float 0~1)
- "reasoning": string
- "key_evidence": string[]
- "severity": object  (key=risk_type, value in {{"low","medium","high"}})
"""),
            ("user", """Please analyze this news article:

Title: {title}
Content: {content}

Provide:
1) risks
2) confidence
3) reasoning
4) key_evidence
5) severity

Make sure:
- The JSON object is valid.
- Risk type keys strictly come from: ["credit_risk","market_risk","operational_risk","fraud_risk"].
- confidence values are floats in [0,1].
- severity values are one of: low, medium, high."""),
        ])

        self.analysis_chain = self.risk_analysis_prompt | self.llm

    def _parse_json(self, resp) -> dict:
        raw = getattr(resp, "content", resp)
        try:
            if isinstance(raw, str):
                try:
                    return json.loads(raw)
                except Exception:
                    start = raw.find("{")
                    end = raw.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        return json.loads(raw[start : end + 1])
                    return {}
            if isinstance(raw, dict):
                return raw
            text = str(raw)
            try:
                return json.loads(text)
            except Exception:
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    return json.loads(text[start : end + 1])
                return {}
        except Exception:
            return {}

    def _normalize(self, data: dict) -> RiskAssessment:
        risks = data.get("risks") or []
        if not isinstance(risks, list):
            risks = []
        risks = [r for r in risks if r in ALLOWED_RISKS]

        conf = data.get("confidence") or {}
        if not isinstance(conf, dict):
            conf = {}
        conf = {
            k: float(max(0.0, min(1.0, v)))
            for k, v in conf.items()
            if k in ALLOWED_RISKS and isinstance(v, (int, float))
        }

        reasoning = data.get("reasoning") or ""
        if not isinstance(reasoning, str):
            reasoning = str(reasoning)

        key_ev = data.get("key_evidence") or []
        if not isinstance(key_ev, list):
            key_ev = []
        key_ev = [str(x) for x in key_ev]

        return RiskAssessment(
            risk_types=risks,
            confidence_scores=conf,
            reasoning=reasoning.strip(),
            key_indicators=key_ev,
        )

    def analyze_news(self, title: str, content: str, max_chars: int = 8000) -> RiskAssessment:
        try:
            full_content = f"{title}\n\n{content}"
            if len(full_content) > max_chars:
                full_content = full_content[:max_chars] + "..."

            resp = self.analysis_chain.invoke({"title": title, "content": full_content})
            data = self._parse_json(resp)

            if not data or not isinstance(data, dict):
                return RiskAssessment(
                    [],
                    {},
                    "AI returned non-JSON or empty result",
                    [],
                    status="error",
                    error_message="AI returned non-JSON or empty result",
                )

            result = self._normalize(data)

            if not result.risk_types and not result.confidence_scores:
                return RiskAssessment([], {}, "AI analysis returned no risks", [], status="no_risk")

            return result

        except Exception as e:
            error_message = str(e)
            print(f"\nAI分析失敗: {error_message}", flush=True)
            return RiskAssessment(
                [],
                {},
                "AI analysis failed",
                [],
                status="error",
                error_message=error_message,
            )

def build_arg_parser():
    p = argparse.ArgumentParser(description="AI 生成 Guardian 新聞風險標籤")
    p.add_argument("--input_csv", type=str, default="guardian_news_raw.csv",
                   help="輸入 CSV（默認: guardian_news_raw.csv）。需要包含 webTitle / bodyText 欄位；或 title / content 欄位")
    p.add_argument("--output_csv", type=str, default="guardian_news_ai_labeled.csv",
                   help="輸出 CSV（默認: guardian_news_ai_labeled.csv）")
    p.add_argument("--model", type=str, default=BAILIAN_MODEL, help=f"百炼模型（默認: {BAILIAN_MODEL}）")
    p.add_argument("--base_url", type=str, default=BAILIAN_BASE_URL, help="百炼 OpenAI-compatible BASE_URL")
    p.add_argument("--temperature", type=float, default=0.1, help="溫度（默認: 0.1）")
    p.add_argument("--timeout", type=int, default=60, help="請求超時秒數（默認: 60）")
    p.add_argument("--rate_limit_sleep", type=float, default=0.0,
                   help="每次請求後 sleep 秒數（默認: 0，必要時可設 0.2~0.5 以降低速率）")
    p.add_argument("--max_rows", type=int, default=0, help="僅處理前 N 行（0 表示全部）")
    p.add_argument("--overwrite", action="store_true",
                   help="覆蓋已有輸出並從頭開始；默認會讀取 output_csv 並斷點續跑")
    return p

def make_result_row(row, idx: int, title: str, assessment: RiskAssessment) -> Dict:
    result_row = dict(row)
    result_row["_source_row_index"] = idx
    result_row["ai_risk_types"] = ",".join(assessment.risk_types)
    result_row["ai_confidence_scores"] = json.dumps(assessment.confidence_scores, ensure_ascii=False)
    result_row["ai_reasoning"] = assessment.reasoning
    result_row["ai_key_evidence"] = json.dumps(assessment.key_indicators, ensure_ascii=False)
    result_row["ai_status"] = assessment.status
    result_row["ai_error_message"] = assessment.error_message
    result_row["risk_count"] = len(assessment.risk_types)
    result_row["has_credit_risk"] = "credit_risk" in assessment.risk_types
    result_row["has_market_risk"] = "market_risk" in assessment.risk_types
    result_row["has_operational_risk"] = "operational_risk" in assessment.risk_types
    result_row["has_fraud_risk"] = "fraud_risk" in assessment.risk_types
    return result_row

def append_result_row(output_csv: Path, result_row: Dict):
    write_header = not output_csv.exists() or output_csv.stat().st_size == 0
    pd.DataFrame([result_row]).to_csv(
        output_csv,
        mode="a",
        header=write_header,
        index=False,
    )

def load_processed_keys(output_csv: Path) -> set[str]:
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return set()

    existing = pd.read_csv(output_csv, usecols=lambda c: c in {"id", "_source_row_index"})
    processed = set()
    if "id" in existing.columns:
        processed.update(existing["id"].dropna().astype(str))
    if "_source_row_index" in existing.columns:
        processed.update("row_index:" + existing["_source_row_index"].dropna().astype(int).astype(str))
    return processed

def row_key(row, idx: int) -> str:
    value = row.get("id")
    if pd.notna(value) and str(value).strip():
        return str(value)
    return f"row_index:{idx}"

def print_final_summary(output_csv: Path):
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        print("沒有可匯總的輸出文件。")
        return

    result_df = pd.read_csv(output_csv)
    print(f"AI 標籤生成完成/已保存至: {output_csv}")
    print(f"當前輸出共有 {len(result_df)} 篇新聞")

def main():
    api_key = get_bailian_api_key()

    args = build_arg_parser().parse_args()

    labeler = AIBasedRiskLabeler(
        api_key=api_key,
        model_name=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        timeout=args.timeout,
    )

    output_csv = Path(args.output_csv)
    if args.overwrite and output_csv.exists():
        output_csv.unlink()

    processed_keys = load_processed_keys(output_csv)

    df = pd.read_csv(args.input_csv)
    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows)

    title_cols = ["webTitle", "title"]
    content_cols = ["bodyText", "content", "text"]

    def pick_first(row, candidates):
        for c in candidates:
            if c in row and pd.notna(row[c]) and str(row[c]).strip():
                return str(row[c])
        return ""

    total = len(df)
    skipped = 0
    for idx, row in df.iterrows():
        key = row_key(row, idx)
        if key in processed_keys:
            skipped += 1
            continue

        title = pick_first(row, title_cols)
        content = pick_first(row, content_cols)

        if not content:
            assessment = RiskAssessment(
                [],
                {},
                "No content to analyze",
                [],
                status="error",
                error_message="No content to analyze",
            )
            result_row = make_result_row(row, idx, title, assessment)
            append_result_row(output_csv, result_row)
            processed_keys.add(key)
            continue

        print(f"\r分析進度: {idx+1}/{total} | 當前: {title[:50]}...", end="", flush=True)

        assessment = labeler.analyze_news(title, content)

        result_row = make_result_row(row, idx, title, assessment)
        append_result_row(output_csv, result_row)
        processed_keys.add(key)

        if args.rate_limit_sleep > 0:
            sleep(args.rate_limit_sleep)

    if skipped:
        print(f"\n斷點續跑：跳過已處理 {skipped} 篇新聞")

    print("\n保存結果完成，開始匯總...")
    print_final_summary(output_csv)

    result_df = pd.read_csv(output_csv) if output_csv.exists() else pd.DataFrame()
    if len(result_df) and "ai_risk_types" in result_df.columns:
        from collections import Counter
        all_risks = []
        for risks in result_df["ai_risk_types"].dropna():
            if risks:
                all_risks.extend([r for r in risks.split(",") if r])

        risk_counts = Counter(all_risks)
        if risk_counts:
            print("\n風險類型分佈:")
            for risk_type, count in risk_counts.most_common():
                print(f"  {risk_type}: {count} 篇新聞")

    if len(result_df) and "ai_status" in result_df.columns:
        error_df = result_df[result_df["ai_status"].eq("error")]
        print(f"\n異常記錄數: {len(error_df)}")
        if len(error_df):
            print("異常記錄列表（最多顯示前 50 條）:")
            for _, row in error_df.head(50).iterrows():
                source_idx = row.get("_source_row_index", "")
                title = row.get("webTitle") or row.get("title") or ""
                error = row.get("ai_error_message", "")
                print(f"  row={source_idx} | title={str(title)[:80]} | error={str(error)[:200]}")

if __name__ == "__main__":
    main()

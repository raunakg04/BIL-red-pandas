"""Suggest alternative collateral assets from retrieved company context."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from retrieval.provider import create_chat_llm
from retrieval.retriever import get_company_context

load_dotenv(override=True)

DEFAULT_LLM_MODEL = "gpt-5.4-mini"
COMPANY_HEADER_RE = re.compile(r"^C\d{3}\s+[—-]\s+(.+)$")
ASSET_LINE_RE = re.compile(
    r"^(?P<asset>.+?)\s+(?P<value>\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s+(?P<currency>[A-Z]{3})$"
)
VALUE_ONLY_RE = re.compile(r"^\d{1,3}(?:,\d{3})*(?:\.\d+)?$")
CURRENCY_ONLY_RE = re.compile(r"^[A-Z]{3}$")


@dataclass(frozen=True)
class AssetCandidate:
    """Structured asset candidate parsed from retrieved context."""

    name: str
    value: float
    currency: str


def _build_explanation_prompt(
    company_name: str,
    loan_value: float,
    current_asset: str,
    current_asset_value: float,
    threshold: float,
    suggested_asset: AssetCandidate,
) -> str:
    # Only the already-selected asset is referenced by the answer - the full
    # parsed_assets list used to be dumped in here too, but the model never
    # needed it (selection already happened deterministically in Python) and
    # it was pure wasted input tokens, especially for companies with many
    # parsed assets on record.
    selected_json = json.dumps(
        {
            "asset": suggested_asset.name,
            "asset_value": suggested_asset.value,
            "currency": suggested_asset.currency,
        },
        ensure_ascii=True,
    )
    shortfall = max(threshold - current_asset_value, 0.0)

    return f"""
You are a compliance reviewer assistant.

Write a short, factual explanation (1-2 sentences, under 50 words) for this
deterministic collateral decision. Do not invent any numbers or assets.
State: the loan value, the required collateral threshold (50% of the loan
value), why the current asset falls short (its value and the shortfall
amount), and the recommended alternative asset and its value.

- company_name: {company_name}
- loan_value: {loan_value:.6f}
- current_asset: {current_asset}
- current_asset_value: {current_asset_value:.6f}
- required_threshold: {threshold:.6f}
- shortfall: {shortfall:.6f}
- selected_asset: {selected_json}

Return STRICT JSON only:
{{"reason": "..."}}
""".strip()


def _safe_json_parse(content: str) -> Dict[str, Any]:
    """Best-effort JSON parse from model output."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    return json.loads(cleaned)


def _normalize_name(text: str) -> str:
    """Normalize asset names for deterministic matching."""
    return re.sub(r"\s+", " ", text.strip()).lower()


def _parse_assets_from_context(company_name: str, context: str) -> List[AssetCandidate]:
    """Parse pledgeable assets from retrieved context using deterministic regex rules."""
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    assets: List[AssetCandidate] = []
    in_target_company_block = False
    in_assets_section = False
    target_normalized = _normalize_name(company_name)

    i = 0
    while i < len(lines):
        line = lines[i]
        # Skip retrieval metadata markers.
        if line.startswith("[Chunk "):
            i += 1
            continue

        header_match = COMPANY_HEADER_RE.match(line)
        if header_match:
            header_company = _normalize_name(header_match.group(1))
            in_target_company_block = header_company == target_normalized
            in_assets_section = False
            i += 1
            continue

        if not in_target_company_block:
            i += 1
            continue

        lower_line = line.lower()
        if lower_line == "pledgeable assets":
            in_assets_section = True
            i += 1
            continue

        if lower_line.startswith("related entities"):
            in_assets_section = False
            i += 1
            continue

        if not in_assets_section:
            i += 1
            continue

        # Skip asset table header rows.
        if line in {"Asset", "Value", "Currency", "Asset Value Currency"}:
            i += 1
            continue

        asset_match = ASSET_LINE_RE.match(line)
        if asset_match:
            value_text = asset_match.group("value").replace(",", "")
            try:
                value = float(value_text)
            except ValueError:
                i += 1
                continue
            assets.append(
                AssetCandidate(
                    name=asset_match.group("asset").strip(),
                    value=value,
                    currency=asset_match.group("currency"),
                )
            )
            i += 1
            continue

        # Primary table format from PDF chunks: asset line, value line, currency line.
        if i + 2 < len(lines):
            candidate_asset = line
            candidate_value = lines[i + 1]
            candidate_currency = lines[i + 2]
            if VALUE_ONLY_RE.match(candidate_value) and CURRENCY_ONLY_RE.match(
                candidate_currency
            ):
                assets.append(
                    AssetCandidate(
                        name=candidate_asset,
                        value=float(candidate_value.replace(",", "")),
                        currency=candidate_currency,
                    )
                )
                i += 3
                continue

        i += 1

    # Fallback: if chunk boundaries removed company headers, parse all asset-like lines.
    if not assets:
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("[Chunk "):
                i += 1
                continue

            if line in {"Asset", "Value", "Currency", "Asset Value Currency"}:
                i += 1
                continue

            asset_match = ASSET_LINE_RE.match(line)
            if asset_match:
                value_text = asset_match.group("value").replace(",", "")
                try:
                    value = float(value_text)
                except ValueError:
                    i += 1
                    continue
                assets.append(
                    AssetCandidate(
                        name=asset_match.group("asset").strip(),
                        value=value,
                        currency=asset_match.group("currency"),
                    )
                )
                i += 1
                continue

            if i + 2 < len(lines):
                candidate_asset = line
                candidate_value = lines[i + 1]
                candidate_currency = lines[i + 2]
                if VALUE_ONLY_RE.match(candidate_value) and CURRENCY_ONLY_RE.match(
                    candidate_currency
                ):
                    assets.append(
                        AssetCandidate(
                            name=candidate_asset,
                            value=float(candidate_value.replace(",", "")),
                            currency=candidate_currency,
                        )
                    )
                    i += 3
                    continue

            i += 1

    # Deduplicate by normalized name and keep the maximum value for stability.
    dedup: Dict[str, AssetCandidate] = {}
    for asset in assets:
        key = _normalize_name(asset.name)
        existing = dedup.get(key)
        if existing is None or asset.value > existing.value:
            dedup[key] = asset

    return list(dedup.values())


def _choose_best_asset(
    all_assets: List[AssetCandidate],
    threshold: float,
    current_asset: str,
) -> Optional[AssetCandidate]:
    """Deterministically choose the best qualifying asset using Python rules."""
    qualifying = [asset for asset in all_assets if asset.value >= threshold]
    if not qualifying:
        return None

    current_normalized = _normalize_name(current_asset)
    alternative_qualifying = [
        asset for asset in qualifying if _normalize_name(asset.name) != current_normalized
    ]

    pool = alternative_qualifying if alternative_qualifying else qualifying
    # Stable ordering: highest value first, then alphabetical name.
    return sorted(pool, key=lambda asset: (-asset.value, asset.name.lower()))[0]


def _default_reason(
    company_name: str,
    loan_value: float,
    current_asset: str,
    current_asset_value: float,
    suggested_asset: Optional[AssetCandidate],
    threshold: float,
    all_assets: List[AssetCandidate],
) -> str:
    """Deterministic fallback reason used when LLM explanation is unavailable."""
    shortfall = max(threshold - current_asset_value, 0.0)
    shortfall_clause = (
        f"This loan is worth {loan_value:,.2f}, which requires at least {threshold:,.2f} "
        f"(50%) in pledged collateral. The current asset, {current_asset} "
        f"({current_asset_value:,.2f}), falls short by {shortfall:,.2f}."
    )

    if suggested_asset is None:
        if not all_assets:
            return f"{shortfall_clause} No parseable assets were found in {company_name}'s records, so no alternative can be suggested."
        return (
            f"{shortfall_clause} No asset in {company_name}'s records meets the required "
            f"threshold of {threshold:,.2f}, so no substitution can be suggested."
        )
    return (
        f"{shortfall_clause} {company_name}'s records show {suggested_asset.name} "
        f"({suggested_asset.value:,.2f}) meets the requirement, so it is recommended as a "
        "replacement."
    )


def _generate_reason_with_llm(
    company_name: str,
    loan_value: float,
    current_asset: str,
    current_asset_value: float,
    threshold: float,
    all_assets: List[AssetCandidate],
    suggested_asset: Optional[AssetCandidate],
) -> str:
    """Use LLM only for human-readable explanation text.

    Only called when a qualifying alternative asset was actually found -
    the no_match case has nothing for the model to add (there's no asset to
    describe), so callers should skip this entirely and use
    _default_reason() directly for that branch instead of paying for an LLM
    call with no informational upside.
    """
    if suggested_asset is None:
        return _default_reason(
            company_name=company_name,
            loan_value=loan_value,
            current_asset=current_asset,
            current_asset_value=current_asset_value,
            suggested_asset=suggested_asset,
            threshold=threshold,
            all_assets=all_assets,
        )

    try:
        # max_tokens keeps the completion (and its cost) bounded - the task
        # is always a single short sentence, never open-ended generation.
        llm = create_chat_llm(default_model=DEFAULT_LLM_MODEL, temperature=0, max_tokens=120)
        prompt = _build_explanation_prompt(
            company_name=company_name,
            loan_value=loan_value,
            current_asset=current_asset,
            current_asset_value=current_asset_value,
            threshold=threshold,
            suggested_asset=suggested_asset,
        )
        response = llm.invoke(prompt)
        parsed = _safe_json_parse(response.content)
        reason = str(parsed.get("reason", "")).strip()
        if reason:
            return reason
    except Exception:
        pass

    return _default_reason(
        company_name=company_name,
        loan_value=loan_value,
        current_asset=current_asset,
        current_asset_value=current_asset_value,
        suggested_asset=suggested_asset,
        threshold=threshold,
        all_assets=all_assets,
    )


def suggest_asset(
    company_name: str,
    loan_value: float,
    current_asset: str,
    current_asset_value: float = 0.0,
    retrieved_context: str | None = None,
) -> Dict[str, Any]:
    """Suggest an alternative asset using deterministic selection rules in Python."""
    threshold = 0.5 * float(loan_value)
    context = retrieved_context or get_company_context(company_name)

    all_assets = _parse_assets_from_context(company_name, context)
    best_asset = _choose_best_asset(all_assets, threshold, current_asset)
    reason = _generate_reason_with_llm(
        company_name=company_name,
        loan_value=float(loan_value),
        current_asset=current_asset,
        current_asset_value=float(current_asset_value),
        threshold=threshold,
        all_assets=all_assets,
        suggested_asset=best_asset,
    )

    if best_asset is None:
        return {
            "status": "no_match",
            "reason": reason or "No suitable collateral available.",
        }

    return {
        "status": "success",
        "suggested_asset": best_asset.name,
        "asset_value": best_asset.value,
        "reason": reason,
    }

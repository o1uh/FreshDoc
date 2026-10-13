from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

MODEL = "deepseek-flash"
MODEL_VERSION = "DeepSeek-V4.1-Flash"
BASE_URL_DEFAULT = "https://api.deepseek.com"
DOCS_CHECKED = "2026-10-13"

# Цены за 1M токенов, USD (страница pricing от 13.10.2026).
# Off-peak = половина peak. Уникальные промпты идут в основном как cache miss.
PRICE_IN_MISS_PEAK = 0.30
PRICE_IN_MISS_OFF = 0.15
PRICE_IN_HIT_PEAK = 0.006
PRICE_IN_HIT_OFF = 0.003
PRICE_OUT_PEAK = 1.20
PRICE_OUT_OFF = 0.60

# Пиковые часы UTC (пн–пт, без гос. праздников КНР; остальное — off-peak).
PEAK_UTC = "Mon-Fri 01:00-04:00, 06:00-10:00 UTC"
# Та же сетка в +7 (Красноярск): 08:00-11:00 и 13:00-17:00 пн–пт.
PEAK_PLUS7 = "Mon-Fri 08:00-11:00, 13:00-17:00 (+7 UTC)"

# Факт 13.10 по usage.csv (2100 запросов, thinking=disabled): вход ~560 средняя,
# completion ~390 (P1 421 / P2 362 / P3 147; отдельные ответы упираются в max_tokens=1200).
AVG_IN_TOKENS = 560
AVG_COMP_TOKENS = 400
COMP_TOKENS_RANGE = (100, 1200)


@dataclass(frozen=True)
class DeepSeekParams:
    model: str = MODEL
    thinking: str = "disabled"  # 13.10: thinking=enabled давал пустой content (reasoning съедал весь max_tokens) — возврат к R7 (non-thinking)
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 1200  # был 8192 при thinking; без thinking ответ ~300-700 ток.


def load_api_key(env: dict[str, str] | None = None, dotenv_path: Path | None = None) -> str:
    """Ключ из окружения DEEPSEEK_API_KEY (опционально — из .env-файла, не в git)."""
    source = dict(env) if env is not None else dict(os.environ)
    key = (source.get("DEEPSEEK_API_KEY") or "").strip()
    if key:
        return key
    candidates = [dotenv_path] if dotenv_path else [Path(".env"), Path(__file__).resolve().parent.parent.parent / ".env"]
    for path in candidates:
        if path is not None and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("DEEPSEEK_API_KEY="):
                    value = line.split("=", 1)[1].strip().strip("\"'")
                    if value:
                        return value
    raise RuntimeError("Нет DEEPSEEK_API_KEY: задайте переменную окружения или строку в .env (файл не коммитится)")


def split_template(template: str) -> tuple[str, str]:
    """Разбить одиночный промпт (SYSTEM:/USER:) на (system, user_шаблон)."""
    marker = "\nUSER:"
    pos = template.find(marker)
    if pos < 0:
        raise ValueError("В промпте нет раздела USER:")
    system_raw, user_tpl = template[:pos], template[pos + len(marker):]
    system = system_raw.strip()
    if system.upper().startswith("SYSTEM:"):
        system = system[len("SYSTEM:"):].strip()
    if "{function_source}" not in user_tpl:
        raise ValueError("В USER-части нет {function_source}")
    return system, user_tpl.strip()


def build_messages(task: str, function_source: str, prompts_dir: Path) -> list[dict[str, str]]:
    """Сообщения для Chat Completions: system + user с подставленной функцией."""
    names = {"p1": "p1_behavior.txt", "p2": "p2_neutral.txt"}
    if task not in names:
        raise ValueError(f"Задача {task!r}: для API-партии допустимы только p1/p2")
    template = (prompts_dir / names[task]).read_text(encoding="utf-8")
    system, user_tpl = split_template(template)
    user = user_tpl.replace("{function_source}", function_source.strip())
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def request_key(model: str, params: DeepSeekParams, messages: list[dict[str, str]]) -> str:
    payload = json.dumps({"model": model, "params": params.__dict__, "messages": messages},
                         ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _normalize_usage(raw: dict[str, Any] | None) -> dict[str, int]:
    raw = raw or {}
    details = raw.get("completion_tokens_details") or {}
    prompt_details = raw.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": int(raw.get("prompt_tokens") or 0),
        "prompt_cache_hit_tokens": int(raw.get("prompt_cache_hit_tokens", prompt_details.get("cached_tokens", 0)) or 0),
        "prompt_cache_miss_tokens": int(raw.get("prompt_cache_miss_tokens") or 0),
        "completion_tokens": int(raw.get("completion_tokens") or 0),
        "reasoning_tokens": int(raw.get("reasoning_tokens", details.get("reasoning_tokens", 0)) or 0),
        "total_tokens": int(raw.get("total_tokens") or 0),
    }


def call_deepseek(
    messages: list[dict[str, str]],
    api_key: str,
    base_url: str = BASE_URL_DEFAULT,
    params: DeepSeekParams | None = None,
    cache_dir: Path | None = None,
    timeout: int = 300,
    max_retries: int = 3,
    retry_sleep: float = 2.0,
) -> dict[str, Any]:
    """Один вызов chat/completions с дисковым кэшем и 3 повторами.

    Кэш: ключ — хэш (модель, параметры, сообщения). Повторный запуск ключ не тратит.
    Пустой content (известная особенность JSON-режима) — повод для повтора.
    Ключ на диск никогда не пишется.
    """
    params = params or DeepSeekParams()
    key = request_key(params.model, params, messages)
    if cache_dir is not None:
        cache_file = cache_dir / f"ds_{key}.json"
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            cached["cached"] = True
            cached["key"] = key
            return cached
    else:
        cache_file = None

    body = {
        "model": params.model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "thinking": {"type": params.thinking},
        "temperature": params.temperature,
        "top_p": params.top_p,
        "max_tokens": params.max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = base_url.rstrip("/") + "/chat/completions"

    last_error: str | None = None
    attempts = 0
    t0 = time.time()
    for attempt in range(1, max_retries + 1):
        attempts = attempt
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        except requests.RequestException as exc:
            last_error = f"transport: {exc}"
            time.sleep(retry_sleep * 2 ** (attempt - 1))
            continue
        if resp.status_code in (429, 500, 502, 503, 504):
            last_error = f"http_{resp.status_code}: {resp.text[:200]}"
            time.sleep(retry_sleep * 2 ** (attempt - 1))
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"DeepSeek HTTP {resp.status_code}: {resp.text[:500]}")
        try:
            data = resp.json()
        except ValueError:
            last_error = f"bad_json_body: {resp.text[:200]}"
            time.sleep(retry_sleep * 2 ** (attempt - 1))
            continue
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError):
            last_error = f"bad_schema: {str(data)[:200]}"
            time.sleep(retry_sleep * 2 ** (attempt - 1))
            continue
        if not isinstance(content, str) or not content.strip():
            last_error = "empty_content (особенность JSON-режима, повтор)"
            time.sleep(retry_sleep * 2 ** (attempt - 1))
            continue
        result = {
            "key": key,
            "model": data.get("model", params.model),
            "content": content,
            "finish_reason": finish_reason,
            "usage": _normalize_usage(data.get("usage")),
            "latency_s": round(time.time() - t0, 2),
            "attempts": attempts,
            "cached": False,
        }
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, cache_file)
        return result
    raise RuntimeError(f"DeepSeek: {max_retries} попыток исчерпаны ({last_error})")


def cost_usd(n_requests: int, avg_in: float, avg_comp: float, peak: bool) -> float:
    """Стоимость партии в USD при средних токенах на запрос (вход — как cache miss)."""
    in_price = PRICE_IN_MISS_PEAK if peak else PRICE_IN_MISS_OFF
    out_price = PRICE_OUT_PEAK if peak else PRICE_OUT_OFF
    return n_requests * (avg_in * in_price + avg_comp * out_price) / 1_000_000


def cost_table(n_requests_list: list[int], avg_in: float = AVG_IN_TOKENS,
               avg_comp: float = AVG_COMP_TOKENS) -> list[dict[str, Any]]:
    rows = []
    for n in n_requests_list:
        rows.append({
            "n_requests": n,
            "usd_offpeak": round(cost_usd(n, avg_in, avg_comp, peak=False), 3),
            "usd_peak": round(cost_usd(n, avg_in, avg_comp, peak=True), 3),
        })
    return rows

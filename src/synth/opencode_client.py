"""FD-22 (API-вариант): клиент Go-подписки opencode для судьи M3.

Endpoint: ``POST {base_url}/anthropic/v1/messages`` (Go Inference API,
Claude обслуживается семейством Anthropic Messages; сверено с доками 19.10.2026).
Оплата — из месячной квоты Go (Haiku 5.5: $15 на Go / $60 на Go Plus; наш
прогон ~$0.05), НЕ с credits-баланса: без префикса ``/go/`` тот же ключ даёт
402 Insufficient account funds.
Аутентификация: Bearer Go API key (``OPENCODE_API_KEY`` в окружении или в
``.env`` — файл не коммитится; ключ на диск никогда не пишется).

Параметры судьи — строго по R10: ``temperature=0.0``, ``max_tokens=120``,
thinking выключен (ответ — один JSON + одно предложение; thinking съедал бы
лимит, как в FD-11). Оговорка: effort-вариант ``max`` из AA-замеров — сущность
агента opencode, в HTTP-контракте Inference API его нет; прогон идёт базовой
моделью ``claude-haiku-5-5`` с temp 0. R7 соблюдено: генератор синтетики —
DeepSeek, судья — Claude.

Кэш: ключ — хэш (модель, параметры, сообщения); повторный запуск ключ не тратит.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

MODEL = "claude-haiku-5-5"
MODEL_LABEL = "Claude Haiku 5.5"
# Go-подписка ($10/$40): модели идут через /inference/go/*, оплата — из месячной
# квоты (Haiku: $15 на Go, $60 на Go Plus; наш прогон ~$0.05), а НЕ с credits-баланса.
# Без /go/ тот же ключ даёт 402 Insufficient account funds (проверено 19.10.2026).
BASE_URL_DEFAULT = "https://opencode.ai/inference/go"
ANTHROPIC_VERSION = "2023-06-01"

# Прайс opencode API 19.10.2026, USD за 1M (тир ≤100K контекста; наши запросы
# ~2–4k токенов — дешёвый тир; тир >100K — ×5, сюда не попадаем).
PRICE_IN = 0.10
PRICE_OUT = 0.50

# Допущения сметы (пересчитать после smoke по usage.csv):
# вход ~400 ток./запрос (system + diff + doc), выход ~350 (факт smoke 20.10.2026:
# среднее 347; R10 просил 120, но 120 и 500 физически режут валидный JSON).
AVG_IN_TOKENS = 400
AVG_COMP_TOKENS = 350


@dataclass(frozen=True)
class OpenCodeParams:
    model: str = MODEL
    # temperature=None: поле НЕ отправляется. Haiku 5.5 на Go-шлюзе отвечает
    # 400 "`temperature` is deprecated for this model" на любое значение
    # (проверено 20.10.2026). Отклонение от R10 зафиксировано в docs/m3_haiku_run.md.
    temperature: float | None = None
    # R10 просит 120, но эмпирически валидный JSON обрезается даже при stop=end
    # (smoke 20.10.2026: stop=max_tokens, `behavior_chang…` на 120 и обрезка
    # reason на 500). 1000 — с запасом; биллинг по факту, не по лимиту.
    max_tokens: int = 1000


def load_api_key(env: dict[str, str] | None = None, dotenv_path: Path | None = None) -> str:
    """Ключ из окружения OPENCODE_API_KEY (опционально — из .env, не в git)."""
    source = dict(env) if env is not None else dict(os.environ)
    key = (source.get("OPENCODE_API_KEY") or "").strip()
    if key:
        return key
    candidates = [dotenv_path] if dotenv_path else [Path(".env"), Path(__file__).resolve().parent.parent.parent / ".env"]
    for path in candidates:
        if path is not None and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("OPENCODE_API_KEY="):
                    value = line.split("=", 1)[1].strip().strip("\"'")
                    if value:
                        return value
    raise RuntimeError("Нет OPENCODE_API_KEY: задайте переменную окружения или строку в .env (файл не коммитится)")


def split_template(template: str) -> tuple[str, str]:
    """Разбить промпт M3 (SYSTEM:/USER:) на (system, user_шаблон)."""
    marker = "\nUSER:"
    pos = template.find(marker)
    if pos < 0:
        raise ValueError("В промпте M3 нет раздела USER:")
    system_raw, user_tpl = template[:pos], template[pos + len(marker):]
    system = system_raw.strip()
    if system.upper().startswith("SYSTEM:"):
        system = system[len("SYSTEM:"):].strip()
    if "{diff_code}" not in user_tpl or "{doc_old}" not in user_tpl:
        raise ValueError("В USER-части M3 нет {diff_code}/{doc_old}")
    return system, user_tpl.strip()


def build_messages(diff_code: str, doc_old: str, prompts_dir: Path) -> list[dict[str, str]]:
    """Сообщения для Messages API: system отдельно, user — один кейс (точный R10)."""
    template = (prompts_dir / "m3_judge.txt").read_text(encoding="utf-8")
    system, user_tpl = split_template(template)
    user = user_tpl.replace("{diff_code}", diff_code or "").replace("{doc_old}", doc_old or "")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def request_key(model: str, params: OpenCodeParams, messages: list[dict[str, str]]) -> str:
    payload = json.dumps({"model": model, "params": params.__dict__, "messages": messages},
                         ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _normalize_usage(raw: dict[str, Any] | None) -> dict[str, int]:
    raw = raw or {}
    return {
        "prompt_tokens": int(raw.get("input_tokens") or 0),
        "prompt_cache_hit_tokens": int(raw.get("cache_read_input_tokens") or 0),
        "prompt_cache_miss_tokens": int(raw.get("cache_creation_input_tokens") or 0),
        "completion_tokens": int(raw.get("output_tokens") or 0),
        "reasoning_tokens": 0,  # thinking выключен по R10
        "total_tokens": int(raw.get("input_tokens") or 0) + int(raw.get("output_tokens") or 0),
    }


def _extract_text(data: dict[str, Any]) -> tuple[str | None, str]:
    """(text, stop_reason) из ответа Messages API. text=None при пустом контенте."""
    blocks = data.get("content")
    if not isinstance(blocks, list):
        return None, str(data.get("stop_reason") or "")
    parts = [b.get("text", "") for b in blocks
             if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)]
    text = "".join(parts)
    return (text if text.strip() else None), str(data.get("stop_reason") or "")


def call_opencode(
    messages: list[dict[str, str]],
    api_key: str,
    base_url: str = BASE_URL_DEFAULT,
    params: OpenCodeParams | None = None,
    cache_dir: Path | None = None,
    timeout: int = 120,
    max_retries: int = 3,
    retry_sleep: float = 2.0,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Один вызов Messages API с дисковым кэшем и 3 повторами.

    Кэш: ключ — хэш (модель, параметры, сообщения). Повторный запуск ключ
    не тратит. Пустой content — повод для повтора. Ключ на диск не пишется.
    """
    params = params or OpenCodeParams()
    key = request_key(params.model, params, messages)
    cache_file = None
    if cache_dir is not None:
        cache_file = cache_dir / f"oc_{key}.json"
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            cached["cached"] = True
            cached["key"] = key
            return cached

    system = next((m["content"] for m in messages if m.get("role") == "system"), "")
    user_msgs = [{"role": "user", "content": m["content"]}
                 for m in messages if m.get("role") != "system"]
    body: dict[str, Any] = {
        "model": params.model,
        "max_tokens": params.max_tokens,
        "system": system,
        "messages": user_msgs,
    }
    if params.temperature is not None:
        body["temperature"] = params.temperature
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
               "anthropic-version": ANTHROPIC_VERSION,
               "User-Agent": "FreshDoc-m3-judge/1.0"}
    if session_id:
        headers["x-opencode-session"] = session_id
    url = base_url.rstrip("/") + "/anthropic/v1/messages"

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
        if resp.status_code in (429, 500, 502, 503, 504, 529):
            last_error = f"http_{resp.status_code}: {resp.text[:200]}"
            time.sleep(retry_sleep * 2 ** (attempt - 1))
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"OpenCode Inference HTTP {resp.status_code}: {resp.text[:500]}")
        try:
            data = resp.json()
        except ValueError:
            last_error = f"bad_json_body: {resp.text[:200]}"
            time.sleep(retry_sleep * 2 ** (attempt - 1))
            continue
        content, finish_reason = _extract_text(data)
        if content is None:
            last_error = f"empty_content (stop={finish_reason or '?'}, повтор)"
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
    raise RuntimeError(f"OpenCode Inference: {max_retries} попыток исчерпаны ({last_error})")


def cost_usd(n_requests: int, avg_in: float = AVG_IN_TOKENS,
             avg_comp: float = AVG_COMP_TOKENS) -> float:
    """Стоимость партии в USD при средних токенах на запрос (тир ≤100K)."""
    return n_requests * (avg_in * PRICE_IN + avg_comp * PRICE_OUT) / 1_000_000


def cost_table(n_requests_list: list[int], avg_in: float = AVG_IN_TOKENS,
               avg_comp: float = AVG_COMP_TOKENS) -> list[dict[str, Any]]:
    return [{"n_requests": n, "usd": round(cost_usd(n, avg_in, avg_comp), 3)}
            for n in n_requests_list]

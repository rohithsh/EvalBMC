#!/usr/bin/env python3
"""
llm_client.py

Provider adapters and nothing else. No prompts, no program analysis.

Every backend implements one method:

    complete(system, user) -> str

Adding a provider means writing one class and adding one line to PROVIDERS.

KEY ROTATION AND RATE LIMITS
  Any number of API keys can be supplied. Every environment variable matching
  API_KEY, API_KEY_<n> or API_KEY_<name> is collected, in that order, so adding
  a key means adding a line to .env and nothing else.

  On HTTP 429 the client marks that key as limited and retries the same request
  on the next available key. A limited key is not tried again until the clock
  hour changes, because these quotas are hourly: retrying within the same hour
  only earns another 429.

  When every key is limited the client sleeps until the top of the next hour
  and then clears all the marks. Only one thread performs the sleep; the others
  wait on the same barrier, so a pool of workers does not multiply the wait.
  The wait is announced, since an unexplained pause of up to an hour looks like
  a hang.

Standalone check:
    python3 llm_client.py --provider openai \
        --base-url https://chat-ai.academiccloud.de/v1 --model gpt-oss-120b
"""

import argparse
import json
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import requests

from dotenv import load_dotenv
load_dotenv()



class ProviderError(Exception):
    pass


class RateLimited(ProviderError):
    """Every key is rate limited."""


# ------------------------------------------------------------------- keys

KEY_RE = re.compile(r"^API_KEY(?:_(\w+))?$")


def keys_from_env():
    """
    Every API_KEY* variable in the environment, de-duplicated.

    API_KEY comes first, then numbered keys in numeric order, then any named
    ones alphabetically, so the ordering is stable across runs.
    """
    found = []
    for name, value in os.environ.items():
        m = KEY_RE.match(name)
        if not m or not value or not value.strip():
            continue
        suffix = m.group(1)
        if suffix is None:
            rank = (0, 0, "")
        elif suffix.isdigit():
            rank = (1, int(suffix), "")
        else:
            rank = (2, 0, suffix.lower())
        found.append((rank, value.strip(), name))

    found.sort(key=lambda x: x[0])
    out, seen = [], set()
    for _, value, name in found:
        if value not in seen:
            seen.add(value)
            out.append((name, value))
    return out


def seconds_to_next_hour():
    now = datetime.now()
    nxt = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    return max(1.0, (nxt - now).total_seconds())


class KeyRing:
    """
    Rotating set of keys, safe to share across threads.

    A key limited during this clock hour is skipped until the hour rolls over.
    When none are available the ring waits for the next hour boundary once,
    on behalf of every waiting thread, and then clears the marks.
    """

    def __init__(self, keys, verbose=True):
        self.keys = [v for _, v in keys] or ["not-needed"]
        self.names = [n for n, _ in keys] or ["(none)"]
        self.limited = {}                 # index -> hour when it was limited
        self.idx = 0
        self.rotations = 0
        self.waits = 0
        self.verbose = verbose
        self._lock = threading.Lock()
        self._waiting = False
        self._cv = threading.Condition(self._lock)

    @staticmethod
    def _hour():
        return datetime.now().replace(minute=0, second=0, microsecond=0)

    def _available(self):
        """Indices not limited in the current hour (assumes the lock is held)."""
        now = self._hour()
        for i, h in list(self.limited.items()):
            if h < now:
                del self.limited[i]       # a new hour: the quota reset
        return [i for i in range(len(self.keys)) if i not in self.limited]

    def current(self):
        """(key, index). Blocks until the next hour if every key is limited."""
        with self._cv:
            while True:
                avail = self._available()
                if avail:
                    if self.idx not in avail:
                        self.idx = avail[0]
                    return self.keys[self.idx], self.idx

                # every key limited: one thread waits, the rest follow it
                if self._waiting:
                    self._cv.wait(timeout=30)
                    continue

                self._waiting = True
                wait = seconds_to_next_hour()
                self.waits += 1
                if self.verbose:
                    print("[llm_client] all {} key(s) rate limited; waiting "
                          "{:.0f} min for the next hour".format(
                              len(self.keys), wait / 60.0), flush=True)
                self._cv.release()
                try:
                    time.sleep(wait + 5)
                finally:
                    self._cv.acquire()
                self.limited.clear()
                self._waiting = False
                self._cv.notify_all()
                if self.verbose:
                    print("[llm_client] resuming", flush=True)

    def mark_limited(self, idx):
        """Record that this key is limited, and move on."""
        with self._cv:
            self.limited[idx] = self._hour()
            avail = [i for i in range(len(self.keys)) if i not in self.limited]
            if avail:
                self.idx = avail[0]
                self.rotations += 1
            self._cv.notify_all()
            return bool(avail)

    def status(self):
        with self._lock:
            return {"keys": len(self.keys),
                    "limited": len(self.limited),
                    "rotations": self.rotations,
                    "hour_waits": self.waits}


# --------------------------------------------------------------- providers

class Provider(ABC):
    name = "abstract"

    def __init__(self, model, temperature=0.0, max_tokens=1200,
                 timeout=180, extra=None):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra = extra or {}

    @abstractmethod
    def _call(self, system, user):
        """Provider-specific request. Raise on failure."""

    def complete(self, system, user, retries=2, backoff=2.0):
        last = None
        for attempt in range(retries + 1):
            try:
                return self._call(system, user)
            except Exception as e:
                last = e
                if attempt < retries:
                    time.sleep(backoff ** attempt)
        raise ProviderError("{}: {}".format(type(last).__name__, str(last)[:200]))

    def describe(self):
        return {"provider": self.name, "model": self.model,
                "temperature": self.temperature, "max_tokens": self.max_tokens}


class OpenAICompatible(Provider):
    """
    /v1/chat/completions. Works with vLLM, Ollama, OpenAI and most hosted
    providers; switch between them by changing base_url alone.
    """

    name = "openai"

    def __init__(self, model, base_url="https://api.openai.com/v1",
                 keys=None, **kw):
        super().__init__(model, **kw)
        self.base_url = base_url.rstrip("/")
        self.ring = KeyRing(keys or [])

    def _call(self, system, user):
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        payload.update(self.extra)

        while True:
            key, idx = self.ring.current()       # blocks if all are limited
            r = requests.post(
                "{}/chat/completions".format(self.base_url),
                headers={"Authorization": "Bearer {}".format(key),
                         "Content-Type": "application/json"},
                json=payload, timeout=self.timeout)

            if r.status_code == 429:
                self.ring.mark_limited(idx)
                continue                          # next key, or wait

            r.raise_for_status()
            data = r.json()
            choice = data["choices"][0]
            msg = choice.get("message", {}) or {}

            content = msg.get("content")
            if content and content.strip():
                return content

            # Reasoning models (gpt-oss and similar) return a separate
            # 'reasoning' field. When the token budget is spent inside it,
            # 'content' is null and the answer, if any, is at the end of the
            # reasoning.
            for k in ("reasoning", "reasoning_content"):
                alt = msg.get(k)
                if alt and alt.strip():
                    return alt

            raise ProviderError(
                "empty content (finish_reason={}, completion_tokens={})".format(
                    choice.get("finish_reason"),
                    (data.get("usage") or {}).get("completion_tokens")))

    def describe(self):
        d = super().describe()
        d.update(self.ring.status())
        return d


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model, base_url="https://api.anthropic.com/v1",
                 keys=None, **kw):
        super().__init__(model, **kw)
        self.base_url = base_url.rstrip("/")
        self.ring = KeyRing(keys or [])

    def _call(self, system, user):
        payload = {
            "model": self.model, "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": self.temperature, "max_tokens": self.max_tokens,
        }
        payload.update(self.extra)

        while True:
            key, idx = self.ring.current()
            r = requests.post("{}/messages".format(self.base_url),
                              headers={"x-api-key": key,
                                       "anthropic-version": "2023-06-01",
                                       "Content-Type": "application/json"},
                              json=payload, timeout=self.timeout)
            if r.status_code == 429:
                self.ring.mark_limited(idx)
                continue
            r.raise_for_status()
            return "".join(b.get("text", "") for b in r.json().get("content", [])
                           if b.get("type") == "text")

    def describe(self):
        d = super().describe()
        d.update(self.ring.status())
        return d


class EchoProvider(Provider):
    """No inference. Returns a fixed reply so prompts can be inspected."""

    name = "echo"

    def _call(self, system, user):
        return json.dumps({"kind": "unbounded",
                           "reasoning": "echo backend, no inference performed"})


PROVIDERS = {
    "openai": OpenAICompatible,
    "anthropic": AnthropicProvider,
    "echo": EchoProvider,
}


def add_provider_args(ap):
    g = ap.add_argument_group("model")
    g.add_argument("--provider", choices=list(PROVIDERS), default="openai")
    g.add_argument("--model", default=None)
    g.add_argument("--base-url", default=None)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--max-tokens", type=int, default=1200)
    g.add_argument("--request-timeout", type=int, default=180)
    g.add_argument("--extra-json", default=None,
                   help="JSON merged into the request body (provider options)")
    return ap


def build_provider(a):
    if a.provider != "echo" and not a.model:
        raise SystemExit("--model is required for provider '{}'".format(a.provider))

    kw = {"temperature": a.temperature, "max_tokens": a.max_tokens,
          "timeout": a.request_timeout}
    if getattr(a, "extra_json", None):
        kw["extra"] = json.loads(a.extra_json)

    if a.provider == "echo":
        return EchoProvider(a.model or "echo", **kw)

    keys = keys_from_env()
    if not keys:
        print("[llm_client] no API_KEY* variables found in the environment")
    else:
        print("[llm_client] {} key(s): {}".format(
            len(keys), ", ".join(n for n, _ in keys)))

    if a.provider == "openai":
        return OpenAICompatible(a.model,
                                base_url=a.base_url or "https://api.openai.com/v1",
                                keys=keys, **kw)
    if a.provider == "anthropic":
        return AnthropicProvider(a.model,
                                 base_url=a.base_url or "https://api.anthropic.com/v1",
                                 keys=keys, **kw)
    raise SystemExit("unknown provider: {}".format(a.provider))


def main():
    ap = argparse.ArgumentParser(description="check that a provider responds")
    add_provider_args(ap)
    a = ap.parse_args()
    p = build_provider(a)
    print("provider:", p.describe())
    t0 = time.time()
    reply = p.complete("Answer with a single word.", "Say OK.")
    print("reply ({:.2f}s): {}".format(time.time() - t0, reply.strip()[:200]))
    print("after call:", p.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
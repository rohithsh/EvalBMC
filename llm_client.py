#!/usr/bin/env python3
"""
llm_client.py

Provider adapters.

Every backend implements one method:

    complete(system, user) -> str

Available clients:
  openai      any endpoint exposing /v1/chat/completions. Covers vLLM, Ollama, OpenAI
  anthropic   Anthropic Messages API.
  echo        returns a fixed reply without contacting anything; to
              exercise the pipeline and inspect prompts for free.

KEY ROTATION
  Several API keys can be supplied. On HTTP 429 (rate limited) the client moves
  to the next key and retries immediately, and STAYS on that key.
  Keys are read from the environment: API_KEY, then API_KEY_BACKUP, then
  API_KEY_2, API_KEY_3, ... in that order. Whichever are set are used.
"""

import argparse
import json
import os
import threading
import time
from abc import ABC, abstractmethod

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class ProviderError(Exception):
    pass


class RateLimited(ProviderError):
    """Every key was rate limited. Retried with a longer backoff."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def keys_from_env(names=("API_KEY", "API_KEY_BACKUP", "API_KEY_2",
                         "API_KEY_3", "API_KEY_4")):
    """Whichever of these are set, in order, de-duplicated."""
    out = []
    for n in names:
        v = os.environ.get(n)
        if v and v.strip() and v.strip() not in out:
            out.append(v.strip())
    return out


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

    def complete(self, system, user, retries=2, backoff=2.0,
                 rate_limit_wait=30.0):
        """
        Call the model, retrying failures. A rate limit waits longer than an
        ordinary error, since retrying a limited endpoint straight away only
        earns another 429.
        """
        last = None
        for attempt in range(retries + 1):
            try:
                return self._call(system, user)
            except RateLimited as e:
                last = e
                if attempt < retries:
                    wait = e.retry_after or (rate_limit_wait * (attempt + 1))
                    time.sleep(wait)
            except Exception as e:
                last = e
                if attempt < retries:
                    time.sleep(backoff ** attempt)
        raise ProviderError("{}: {}".format(type(last).__name__, str(last)[:200]))

    def describe(self):
        return {"provider": self.name, "model": self.model,
                "temperature": self.temperature, "max_tokens": self.max_tokens}


class KeyRing:
    """Rotating set of API keys, safe to share across threads."""

    def __init__(self, keys, retry_primary_after=300.0):
        self.keys = list(keys) or ["not-needed"]
        self.idx = 0
        self.retry_primary_after = retry_primary_after
        self._switched_at = None
        self._lock = threading.Lock()
        self.rotations = 0

    def current(self):
        with self._lock:
            # periodically give the primary another chance
            if (self.idx != 0 and self._switched_at is not None
                    and time.time() - self._switched_at > self.retry_primary_after):
                self.idx = 0
                self._switched_at = None
            return self.keys[self.idx], self.idx

    def rotate(self, from_idx):
        """
        Move past the key that was limited. Returns True if a different key is
        now in use. from_idx guards against two threads rotating for the same
        429 and skipping a key between them.
        """
        with self._lock:
            if from_idx != self.idx:
                return True                      # another thread already moved
            if len(self.keys) == 1:
                return False
            self.idx = (self.idx + 1) % len(self.keys)
            self._switched_at = time.time()
            self.rotations += 1
            return self.idx != from_idx


class OpenAICompatible(Provider):
    """
    /v1/chat/completions. Works with vLLM, Ollama, OpenAI and most hosted
    providers; switch between them by changing base_url alone.
    """

    name = "openai"

    def __init__(self, model, base_url="https://api.openai.com/v1",
                 api_key=None, api_keys=None, retry_primary_after=300.0, **kw):
        super().__init__(model, **kw)
        self.base_url = base_url.rstrip("/")
        keys = [k for k in (api_keys or []) if k]
        if not keys and api_key:
            keys = [api_key]
        if not keys:
            keys = ["not-needed"]          # local servers usually ignore it
        self.ring = KeyRing(keys, retry_primary_after)

    def _post(self, key, payload):
        return requests.post(
            "{}/chat/completions".format(self.base_url),
            headers={"Authorization": "Bearer {}".format(key),
                     "Content-Type": "application/json"},
            json=payload, timeout=self.timeout)

    def _call(self, system, user):
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        payload.update(self.extra)

        tried, retry_after = 0, None
        while tried < len(self.ring.keys):
            key, idx = self.ring.current()
            r = self._post(key, payload)

            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                try:
                    retry_after = float(ra) if ra else retry_after
                except ValueError:
                    pass
                tried += 1
                if not self.ring.rotate(idx):
                    break                    # only one key, nothing to rotate to
                continue

            r.raise_for_status()
            data = r.json()
            choice = data["choices"][0]
            msg = choice.get("message", {}) or {}

            content = msg.get("content")
            if content and content.strip():
                return content

            # Reasoning models (gpt-oss and similar) return a separate
            # 'reasoning' field. When the token budget is spent inside it,
            # 'content' comes back null and the answer, if any, is at the end
            # of the reasoning.
            for k in ("reasoning", "reasoning_content"):
                alt = msg.get(k)
                if alt and alt.strip():
                    return alt

            raise ProviderError(
                "empty content (finish_reason={}, completion_tokens={})".format(
                    choice.get("finish_reason"),
                    (data.get("usage") or {}).get("completion_tokens")))

        raise RateLimited(
            "all {} key(s) rate limited".format(len(self.ring.keys)),
            retry_after=retry_after)

    def describe(self):
        d = super().describe()
        d["keys"] = len(self.ring.keys)
        d["key_rotations"] = self.ring.rotations
        return d


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model, base_url="https://api.anthropic.com/v1",
                 api_key=None, api_keys=None, retry_primary_after=300.0, **kw):
        super().__init__(model, **kw)
        self.base_url = base_url.rstrip("/")
        keys = [k for k in (api_keys or []) if k] or ([api_key] if api_key else [])
        self.ring = KeyRing(keys or [""], retry_primary_after)

    def _call(self, system, user):
        payload = {
            "model": self.model, "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": self.temperature, "max_tokens": self.max_tokens,
        }
        payload.update(self.extra)

        tried, retry_after = 0, None
        while tried < len(self.ring.keys):
            key, idx = self.ring.current()
            r = requests.post("{}/messages".format(self.base_url),
                              headers={"x-api-key": key,
                                       "anthropic-version": "2023-06-01",
                                       "Content-Type": "application/json"},
                              json=payload, timeout=self.timeout)
            if r.status_code == 429:
                ra = r.headers.get("retry-after")
                try:
                    retry_after = float(ra) if ra else retry_after
                except ValueError:
                    pass
                tried += 1
                if not self.ring.rotate(idx):
                    break
                continue
            r.raise_for_status()
            return "".join(b.get("text", "") for b in r.json().get("content", [])
                           if b.get("type") == "text")

        raise RateLimited(
            "all {} key(s) rate limited".format(len(self.ring.keys)),
            retry_after=retry_after)

    def describe(self):
        d = super().describe()
        d["keys"] = len(self.ring.keys)
        d["key_rotations"] = self.ring.rotations
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
    """Shared flags, so every script takes the same model options."""
    g = ap.add_argument_group("model")
    g.add_argument("--provider", choices=list(PROVIDERS), default="openai")
    g.add_argument("--model", default=None)
    g.add_argument("--base-url", default=None)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--max-tokens", type=int, default=1200)
    g.add_argument("--request-timeout", type=int, default=180)
    g.add_argument("--extra-json", default=None,
                   help="JSON merged into the request body (provider options)")
    g.add_argument("--key-retry-primary", type=float, default=300.0,
                   help="seconds before the primary key is tried again after "
                        "rotating away from it")
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
        print("[warn] no API_KEY / API_KEY_BACKUP found in the environment")
    else:
        print("using {} API key(s)".format(len(keys)))

    retry_primary = getattr(a, "key_retry_primary", 300.0)
    if a.provider == "openai":
        return OpenAICompatible(a.model,
                                base_url=a.base_url or "https://api.openai.com/v1",
                                api_keys=keys,
                                retry_primary_after=retry_primary, **kw)
    if a.provider == "anthropic":
        return AnthropicProvider(a.model,
                                 base_url=a.base_url or "https://api.anthropic.com/v1",
                                 api_keys=keys,
                                 retry_primary_after=retry_primary, **kw)
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
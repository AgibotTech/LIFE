"""Minimal OpenAI-compatible chat client with round-robin over several vLLM endpoints."""
import itertools, json, os, threading, time, urllib.request


class Client:
    def __init__(self, base_urls, model, api_key_env='SSS_JUDGE_API_KEY'):
        self.urls = [u.rstrip('/') for u in (base_urls.split(',') if isinstance(base_urls, str) else base_urls) if u]
        self.model, self.key = model, os.environ.get(api_key_env)
        self._cycle, self._lock = itertools.cycle(self.urls), threading.Lock()

    def chat(self, messages, max_tokens, temperature=0.0, retries=4, **extra):
        """-> (text, finish_reason)"""
        body = dict(model=self.model, messages=messages, temperature=temperature, seed=0, top_p=1.0, max_tokens=max_tokens,
                    chat_template_kwargs={'enable_thinking': False}, **extra)
        headers = {'Content-Type': 'application/json'}
        if self.key:
            headers['Authorization'] = 'Bearer ' + self.key
        err = None
        for attempt in range(retries):
            with self._lock:
                url = next(self._cycle)
            try:
                req = urllib.request.Request(url + '/chat/completions', data=json.dumps(body).encode(), headers=headers)
                with urllib.request.urlopen(req, timeout=1800) as r:
                    c = json.load(r)['choices'][0]
                return c['message']['content'], c.get('finish_reason')
            except Exception as e:
                err = e; time.sleep(2 * (attempt + 1))
        raise RuntimeError(f'chat failed on {self.urls}: {err!r}')

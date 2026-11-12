"""MLX backend for the controller: Qwen3-4B-Instruct-2507, 4-bit (mlx-community), greedy decoding."""
import time

MODEL_REPO = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
MODEL_REVISION = "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"   # pinned snapshot


class MLXBackend:
    def __init__(self, repo: str = MODEL_REPO, revision: str = MODEL_REVISION):
        import mlx.core as mx
        from mlx_lm import load
        self.mx = mx
        self.name = f"mlx:{repo}@{revision[:7]}"
        t = time.perf_counter()
        self.model, self.tokenizer = load(repo, revision=revision)
        self.load_seconds = time.perf_counter() - t
        self.repo, self.revision = repo, revision

    def generate(self, system, user, max_new_tokens, temperature, extra_messages=None):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}] + list(extra_messages or [])
        prompt = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        t, text, last = time.perf_counter(), "", None
        for r in stream_generate(self.model, self.tokenizer, prompt, max_tokens=max_new_tokens,
                                 sampler=make_sampler(temp=temperature)):
            text += r.text; last = r
        return {"text": text, "prompt_tokens": last.prompt_tokens, "output_tokens": last.generation_tokens,
                "latency_ms": (time.perf_counter() - t) * 1000, "prompt_tps": last.prompt_tps,
                "generation_tps": last.generation_tps, "finish_reason": last.finish_reason,
                "mlx_peak_memory_gb": self.mx.get_peak_memory() / 1e9}

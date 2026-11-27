"""Vision-language model backends: one image and one prompt in, JSON text out."""

import base64
import time
from dataclasses import asdict, dataclass
from typing import Protocol

import httpx

from aomma.config import VLMConfig
from aomma.jobqueue import JobQueue

# httpx times each phase of a request on its own, so a local question is capped by
# their sum: five minutes, above the longest answer on record bar one 635 s outlier.
TIMEOUT = httpx.Timeout(connect=10, write=20, pool=10, read=260)
TIMEOUT_TOTAL_S = TIMEOUT.connect + TIMEOUT.write + TIMEOUT.pool + TIMEOUT.read
# A question that times out is asked twice more before the stage fails: one retry was
# not always enough (a review's question has timed out on both).
TIMEOUT_RETRIES = 2
# A noise draw's sampling: each gate round is judged against a noise floor, the same build again with the VLM's
# sampling moved. Runs answer at temperature 0, so the same input gives the
# same listing, and SAM 3's detection does not sample: a second build at the same commit would show no spread. A noise
# draw's run names a seed in run.json (`vlm_sampling`), and its local VLM answers at Ollama's own default temperature
# with that seed. Production runs never set it. Checked live: the temperature moves the answers, but Ollama's
# MLX runner does not honour the seed (seed 1 asked twice gave two different lists), so a noise draw is a fresh random
# draw that can't be replayed; the seed only names it.
NOISE_TEMPERATURE = 0.8


@dataclass
class Usage:
    model: str
    input_tokens: int
    output_tokens: int
    seconds: float
    # False for local models, which cost time but not money.
    billed: bool

    def to_dict(self) -> dict:
        return asdict(self)


class VLM(Protocol):
    def ask(
        self, image_jpeg: bytes, prompt: str, schema: dict, label: str = "", ticket: int | None = None
    ) -> tuple[str, Usage]:
        """Return the model's reply, which should be JSON matching `schema`.

        Backends that can constrain output to the schema do so; the rest rely
        on the prompt, so callers must still parse and validate the reply.
        `label` says what the question is for, and `ticket` is a place taken
        earlier in the harness's job queue; both matter only to backends that
        run on the shared model machine.
        """
        ...


class OllamaVLM:
    def __init__(self, url: str, model: str, queue: JobQueue | None = None, run: str = "", sampling: dict | None = None):
        self.url = url.rstrip("/")
        self.model = model
        # Ollama's options: temperature 0, or a noise draw's temperature and seed.
        self.options = {"temperature": 0} | (sampling or {})
        # The harness's one job queue, in which each question waits its turn. None: no waiting.
        self.queue, self.run_name = queue, run

    def ask(
        self, image_jpeg: bytes, prompt: str, schema: dict, label: str = "", ticket: int | None = None
    ) -> tuple[str, Usage]:
        if self.queue is None:
            return self._ask(image_jpeg, prompt)
        with self.queue.turn("vlm", label or "VLM question", self.run_name, ticket):
            return self._ask(image_jpeg, prompt)

    def chat(self, messages: list[dict], tools: list[dict], label: str = "") -> tuple[dict, Usage]:
        """One turn of a conversation in which the model may call `tools`; returns its message.

        `messages` and `tools` are in Ollama's chat format. The turn waits in
        the queue as a question does, so a long conversation is a string of
        short jobs and holds the machine only while the model is answering.
        """
        if self.queue is None:
            return self._post(messages, tools, think=True)
        with self.queue.turn("vlm", label or "VLM turn", self.run_name):
            return self._post(messages, tools, think=True)

    def _ask(self, image_jpeg: bytes, prompt: str) -> tuple[str, Usage]:
        message, usage = self._post(
            [{"role": "user", "content": prompt, "images": [base64.b64encode(image_jpeg).decode()]}]
        )
        return message["content"], usage

    def _post(self, messages: list[dict], tools: list[dict] | None = None, think: bool = False) -> tuple[dict, Usage]:
        start = time.monotonic()
        # No `format` constraint: Ollama's MLX runner rejects structured output.
        body = {
            "model": self.model,
            "stream": False,
            "think": think,
            "options": self.options,
            "messages": messages,
            **({"tools": tools} if tools else {}),
        }
        for attempt in range(1 + TIMEOUT_RETRIES):
            try:
                response = httpx.post(f"{self.url}/api/chat", json=body, timeout=TIMEOUT)
                break
            except httpx.TimeoutException:
                # Ollama sometimes stops answering a question that was waiting when a 3D
                # job took the GPU; the same question asked again is answered at once.
                if attempt == TIMEOUT_RETRIES:
                    raise
                print(f"{self.model}: no answer in {TIMEOUT_TOTAL_S:.0f} s, asking again", flush=True)
        response.raise_for_status()
        data = response.json()
        usage = Usage(
            model=self.model,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            seconds=round(time.monotonic() - start, 1),
            billed=False,
        )
        return data["message"], usage


class AnthropicVLM:
    def __init__(self, model: str, effort: str):
        import anthropic

        self.model = model
        self.effort = effort
        self._client = anthropic.Anthropic()

    def ask(
        self, image_jpeg: bytes, prompt: str, schema: dict, label: str = "", ticket: int | None = None
    ) -> tuple[str, Usage]:
        # A cloud model does not use the shared machine, so it does not queue.
        start = time.monotonic()
        response = self._client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            # If a safety classifier declines, retry server-side on the
            # fallback model Anthropic recommends instead of failing the run.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64.standard_b64encode(image_jpeg).decode(),
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        if response.stop_reason in ("refusal", "max_tokens"):
            raise RuntimeError(f"{self.model} stopped early: {response.stop_reason}")
        u = response.usage
        usage = Usage(
            model=response.model,
            input_tokens=u.input_tokens
            + (u.cache_creation_input_tokens or 0)
            + (u.cache_read_input_tokens or 0),
            output_tokens=u.output_tokens,
            seconds=round(time.monotonic() - start, 1),
            billed=True,
        )
        text = next(b.text for b in response.content if b.type == "text")
        return text, usage


def make_vlm(config: VLMConfig, queue: JobQueue | None = None, run: str = "", sampling: dict | None = None) -> VLM:
    """Build the configured backend. `queue` and `run` put a local model's questions in the harness's job queue;
    `sampling` is a noise draw's temperature and seed (local models only)."""
    if config.backend == "ollama":
        if not config.url:
            raise ValueError("ollama backend needs a `url`")
        return OllamaVLM(config.url, config.model, queue, run, sampling)
    if sampling:
        raise SystemExit(f"{config.model}: a noise draw's seeded sampling needs a local model")
    return AnthropicVLM(config.model, config.effort)


def noise_sampling(seed: int) -> dict:
    """What a noise draw's run.json holds as `vlm_sampling`."""
    return {"temperature": NOISE_TEMPERATURE, "seed": seed}

import httpx
import pytest

from aomma.backends import vlm
from aomma.backends.vlm import OllamaVLM

ANSWER = {"message": {"role": "assistant", "content": "{}"}, "prompt_eval_count": 10, "eval_count": 2}


def _post_with(monkeypatch, replies):
    """Make httpx.post give `replies` in turn: an exception is raised, a dict is answered."""
    calls = []

    def post(url, json, timeout):
        calls.append(timeout)
        reply = replies[len(calls) - 1]
        if isinstance(reply, Exception):
            raise reply
        return httpx.Response(200, json=reply, request=httpx.Request("POST", url))

    monkeypatch.setattr(vlm.httpx, "post", post)
    return calls


def test_a_question_is_capped_at_a_few_minutes_in_all():
    t = vlm.TIMEOUT
    assert t.connect + t.write + t.pool + t.read == vlm.TIMEOUT_TOTAL_S <= 300


def test_a_question_that_times_out_is_asked_again(monkeypatch, capsys):
    calls = _post_with(monkeypatch, [httpx.ReadTimeout("timed out"), ANSWER])

    text, usage = OllamaVLM("http://ollama.test", "qwen").ask(b"jpeg", "prompt", {})

    assert text == "{}"
    assert usage.output_tokens == 2
    assert calls == [vlm.TIMEOUT, vlm.TIMEOUT]
    assert "asking again" in capsys.readouterr().out


def test_a_question_that_times_out_twice_is_answered_the_third_time(monkeypatch):
    calls = _post_with(monkeypatch, [httpx.WriteTimeout("timed out"), httpx.ReadTimeout("timed out"), ANSWER])

    text, _ = OllamaVLM("http://ollama.test", "qwen").ask(b"jpeg", "prompt", {})

    assert text == "{}"
    assert len(calls) == 3


def test_a_question_that_times_out_three_times_fails(monkeypatch):
    timeouts = [httpx.ReadTimeout("timed out")] * 3
    calls = _post_with(monkeypatch, [*timeouts, ANSWER])

    with pytest.raises(httpx.ReadTimeout):
        OllamaVLM("http://ollama.test", "qwen").ask(b"jpeg", "prompt", {})
    assert len(calls) == 3


def test_other_failures_are_not_retried(monkeypatch):
    calls = _post_with(monkeypatch, [httpx.ConnectError("refused"), ANSWER])

    with pytest.raises(httpx.ConnectError):
        OllamaVLM("http://ollama.test", "qwen").ask(b"jpeg", "prompt", {})
    assert len(calls) == 1


def _options_sent(monkeypatch, model):
    sent = []

    def post(url, json, timeout):
        sent.append(json["options"])
        return httpx.Response(200, json=ANSWER, request=httpx.Request("POST", url))

    monkeypatch.setattr(vlm.httpx, "post", post)
    model.ask(b"jpeg", "prompt", {})
    return sent[0]


def test_a_run_answers_at_temperature_zero(monkeypatch):
    assert _options_sent(monkeypatch, OllamaVLM("http://ollama.test", "qwen")) == {"temperature": 0}


def test_a_noise_draw_is_sampled_with_its_seed(monkeypatch):
    model = OllamaVLM("http://ollama.test", "qwen", sampling=vlm.noise_sampling(3))
    assert _options_sent(monkeypatch, model) == {"temperature": vlm.NOISE_TEMPERATURE, "seed": 3}


def test_the_run_manifest_sets_a_noise_draw(tmp_path, monkeypatch):
    import json

    from aomma import cli

    (tmp_path / "run.json").write_text(json.dumps({"vlm_sampling": vlm.noise_sampling(5)}))
    config = cli.load_config(cli.DEFAULT_CONFIG)
    if config.vlms[config.vlm].backend != "ollama":
        pytest.skip("the configured VLM is not local")
    assert cli.vlm_for(config, None, tmp_path).options == {"temperature": vlm.NOISE_TEMPERATURE, "seed": 5}
    (tmp_path / "run.json").write_text("{}")
    assert cli.vlm_for(config, None, tmp_path).options == {"temperature": 0}

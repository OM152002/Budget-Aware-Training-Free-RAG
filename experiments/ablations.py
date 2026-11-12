"""A3 "no adaptive stopping" (EXPERIMENT_PLAN.md §1), as a wrapper around the UNMODIFIED TraversalController.

Each step first makes exactly the frozen M5 call (same prompt, same retry semantics). Only if that yields a valid
STOP does the wrapper send ONE fixed re-prompt in which STOP is not available:
    [system, user(frozen prompt), assistant(<the STOP reply>), user(NO_STOP_MESSAGE)]
and parse it with the frozen parser. A valid EXPLORE is returned (call stats of both calls are summed and the
STOP reply is logged); anything else (STOP again or invalid) is returned as INVALID (error STOP_NOT_AVAILABLE or
the parser's code), so the frozen traverse() applies its single retry; if that retry again ends without a valid
EXPLORE, traversal ends with CONTROLLER_ERROR. Hence trajectories equal M5's up to its first STOP.
"""
from controller import _alias_range, build_prompt, parse_action

NO_STOP_MESSAGE = ("STOP is not available: you must explore one more passage. Only the candidate labels {r} are "
                   'valid actions. Reply with only one JSON object: {{"action":"EXPLORE","node_id":"A<number>"}}')


class NoStopController:
    """`events` logs every suppressed STOP (frozen traverse() call records do not keep decision flags), and
    `extra_calls` counts the re-prompt model calls, which traverse() cannot see (it counts one call per decision).
    Call reset() before each question."""

    def __init__(self, base):
        self.base, self.cfg, self.backend = base, base.cfg, base.backend
        self.reset()

    def reset(self):
        self.events, self.extra_calls = [], 0

    def choose_action(self, query, evidence, candidates, history, remaining_budget, previous_invalid=None):
        d = self.base.choose_action(query, evidence, candidates, history, remaining_budget, previous_invalid)
        if not (d.valid and d.action == "STOP"):
            return d
        self.extra_calls += 1
        system, user, alias_map = build_prompt(query, evidence, candidates, history, remaining_budget, self.cfg)
        extra = [{"role": "assistant", "content": d.raw_output},
                 {"role": "user", "content": NO_STOP_MESSAGE.format(r=_alias_range("A", len(candidates)))}]
        out = self.backend.generate(system, user, self.cfg.max_new_tokens, self.cfg.temperature, extra_messages=extra)
        d2 = parse_action(out["text"], alias_map)
        if d2.valid and d2.action == "STOP":
            d2 = parse_action("", alias_map)                      # placeholder invalid decision ...
            d2.error, d2.raw_output = "STOP_NOT_AVAILABLE", out["text"]   # ... carrying the real reply/code
        d2.prompt_tokens = d.prompt_tokens + out["prompt_tokens"]
        d2.output_tokens = d.output_tokens + out["output_tokens"]
        d2.latency_ms = d.latency_ms + out["latency_ms"]
        d2.flags = list(d2.flags) + ["no_stop_reprompt"]
        d2.backend_stats = {"suppressed_stop_raw_output": d.raw_output, "no_stop_reprompt_calls": 1}
        self.events.append({"history_len": len(history), "remaining_budget": remaining_budget,
                            "retry_attempt": previous_invalid is not None, "suppressed_stop_raw": d.raw_output,
                            "stop_call_tokens": [d.prompt_tokens, d.output_tokens],
                            "reprompt_raw": out["text"], "reprompt_tokens": [out["prompt_tokens"], out["output_tokens"]],
                            "reprompt_result": d2.action if d2.valid else d2.error})
        return d2

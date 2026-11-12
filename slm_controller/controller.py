"""Model-agnostic traversal controller: one structured action per call (PRD §13-17).

Input  X = (query, evidence, candidates, history, remaining_budget); no gold labels ever enter the prompt.
Output exactly one of  {"action":"EXPLORE","node_id":"<candidate id>"}  or  {"action":"STOP","node_id":null}.

Prompt v2 (default): evidence is shown as unlabeled, non-selectable context; candidates use action aliases A1..Am;
the prompt states that only those aliases or STOP are valid and that the budget is a maximum, not a target.
Prompt v1 (accepted earlier pilot, kept for reproducibility): E1..Ek evidence / C1..Cm candidate aliases.
The controller maps a valid alias back to its chunk ID. Validation is strict and explicit:
  MALFORMED_JSON / EMPTY_OUTPUT / MULTIPLE_ACTIONS / INVALID_ACTION / MISSING_NODE_ID / CONFLICTING_STOP /
  VISITED_NODE (an evidence/visited alias) / NODE_NOT_IN_CANDIDATES.
Deterministic fallback for any invalid output: effective action STOP with termination status CONTROLLER_ERROR
(distinct from SLM_STOP); no retry, no reinterpretation.
"""
import json
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Protocol

VALID_ACTIONS = ("EXPLORE", "STOP")


@dataclass
class Passage:
    chunk_id: str
    text: str


@dataclass
class HistoryStep:
    step: int
    action: str                 # EXPLORE / STOP
    chunk_id: Optional[str]


@dataclass
class ControllerConfig:
    max_passage_chars: int = 300
    max_new_tokens: int = 48
    temperature: float = 0.0
    include_history: bool = True
    include_budget: bool = True
    prompt_version: str = "v2"   # v1 = accepted pilot (E/C aliases); v2 = unlabeled evidence, A-aliases, STOP clarification


@dataclass
class Decision:
    action: str                          # parsed action if valid, else "INVALID"
    node_id: Optional[str]               # chunk ID (mapped from alias) for a valid EXPLORE
    valid: bool
    error: Optional[str]                 # validation error code
    effective_action: str                # EXPLORE / STOP after fallback
    termination: Optional[str]           # SLM_STOP / CONTROLLER_ERROR / None
    alias: Optional[str] = None
    raw_output: str = ""
    flags: List[str] = field(default_factory=list)
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    backend_stats: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class Backend(Protocol):
    name: str

    def generate(self, system: str, user: str, max_new_tokens: int, temperature: float) -> dict:
        """-> {"text", "prompt_tokens", "output_tokens", "latency_ms", ...stats}"""


SYSTEM_PROMPT = (
    "You guide step-by-step retrieval over a graph of text passages to answer a question.\n"
    "You see the evidence retrieved so far and candidate passages linked to it.\n"
    "Choose exactly one action:\n"
    "- EXPLORE the single candidate most likely to contain information still needed to answer the question;\n"
    "- STOP if the evidence already suffices or no candidate is likely to help.\n"
    "Each EXPLORE uses one unit of budget.\n"
    'Reply with only one JSON object: {"action":"EXPLORE","node_id":"<candidate id>"} or {"action":"STOP","node_id":null}'
)


SYSTEM_PROMPT_V2 = (
    "You guide step-by-step retrieval over a graph of text passages to answer a question.\n"
    "You see the evidence retrieved so far (context only, it cannot be selected) and candidate passages "
    "labeled A1, A2, ... that are linked to it.\n"
    "Choose exactly one action:\n"
    "- EXPLORE one candidate that is likely to add information still needed to answer the question;\n"
    "- STOP when the evidence already suffices to answer the question, or when none of the candidates "
    "seems likely to add useful evidence.\n"
    "The remaining budget is a maximum, not a target: you do not need to use it.\n"
    "Only the candidate labels shown (A1, A2, ...) or STOP are valid actions.\n"
    'Reply with only one JSON object: {"action":"EXPLORE","node_id":"A<number>"} or {"action":"STOP","node_id":null}'
)


def _alias_range(prefix: str, n: int) -> str:
    return f"{prefix}1" if n == 1 else f"{prefix}1 to {prefix}{n}"


def correction_message(n_candidates: int, version: str = "v2") -> str:
    """Fixed, deterministic correction used for the single retry after an invalid reply."""
    if version == "v1":
        return (f"That reply was invalid. Only the candidate ids {_alias_range('C', n_candidates)} or STOP are valid; "
                "evidence ids cannot be chosen. "
                'Reply with only one JSON object: {"action":"EXPLORE","node_id":"<candidate id>"} or {"action":"STOP","node_id":null}')
    return (f"That reply was invalid. Only the candidate labels {_alias_range('A', n_candidates)} or STOP are valid "
            "actions. "
            'Reply with only one JSON object: {"action":"EXPLORE","node_id":"A<number>"} or {"action":"STOP","node_id":null}')


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rstrip() + " ..."


def build_prompt(query, evidence: List[Passage], candidates: List[Passage], history: List[HistoryStep],
                 remaining_budget: int, cfg: ControllerConfig):
    """Returns (system, user, alias_map). alias_map["candidates"]: selectable alias -> chunk_id;
    alias_map["evidence"]: selectable-looking evidence aliases (v1 only; empty in v2)."""
    if cfg.prompt_version == "v2":
        return _build_prompt_v2(query, evidence, candidates, history, remaining_budget, cfg)
    ev_alias = {f"E{i + 1}": p.chunk_id for i, p in enumerate(evidence)}
    cand_alias = {f"C{i + 1}": p.chunk_id for i, p in enumerate(candidates)}
    by_chunk = {v: k for k, v in {**ev_alias, **cand_alias}.items()}
    lines = [f"Question: {query}"]
    if cfg.include_budget:
        lines.append(f"Remaining budget: {remaining_budget}")
    if cfg.include_history:
        if history:
            hist = "; ".join(f"step {h.step}: {h.action}" + (f" {by_chunk.get(h.chunk_id, '?')}" if h.chunk_id else "")
                             for h in history)
        else:
            hist = "none"
        lines.append(f"History: {hist}")
    lines.append("Evidence:")
    lines += [f"[{a}] {_clip(p.text, cfg.max_passage_chars)}" for a, p in zip(ev_alias, evidence)]
    lines.append("Candidates:")
    lines += [f"[{a}] {_clip(p.text, cfg.max_passage_chars)}" for a, p in zip(cand_alias, candidates)]
    lines.append("Answer with the JSON object only.")
    return SYSTEM_PROMPT, "\n".join(lines), {"evidence": ev_alias, "candidates": cand_alias}


def _build_prompt_v2(query, evidence, candidates, history, remaining_budget, cfg):
    """Evidence as plain unlabeled bullets; candidates as A1..Am; history refers to passages by a short text snippet."""
    cand_alias = {f"A{i + 1}": p.chunk_id for i, p in enumerate(candidates)}
    text_of = {p.chunk_id: p.text for p in evidence}
    lines = [f"Question: {query}"]
    if cfg.include_budget:
        lines.append(f"Remaining budget (maximum further explorations): {remaining_budget}")
    if cfg.include_history:
        if history:
            hist = "; ".join(
                f"step {h.step}: {h.action}" + (f' "{_clip(text_of.get(h.chunk_id, ""), 60)}"' if h.chunk_id else "")
                for h in history)
        else:
            hist = "none"
        lines.append(f"History: {hist}")
    lines.append("Evidence so far (context only, not selectable):")
    lines += [f"- {_clip(p.text, cfg.max_passage_chars)}" for p in evidence]
    lines.append(f"Candidates (valid actions: {_alias_range('A', len(candidates))} or STOP):")
    lines += [f"[{a}] {_clip(p.text, cfg.max_passage_chars)}" for a, p in zip(cand_alias, candidates)]
    lines.append("Answer with the JSON object only.")
    return SYSTEM_PROMPT_V2, "\n".join(lines), {"evidence": {}, "candidates": cand_alias}


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)


def parse_action(raw: str, alias_map: dict) -> Decision:
    """Strict parse + validation of one model output against the supplied alias map."""
    flags = []
    text = (raw or "").strip()

    def bad(code):
        return Decision("INVALID", None, False, code, "STOP", "CONTROLLER_ERROR", raw_output=raw, flags=flags)

    if not text:
        return bad("EMPTY_OUTPUT")
    m = _FENCE.match(text)
    if m:
        text = m.group(1).strip(); flags.append("code_fence")
    dec, objs, i = json.JSONDecoder(), [], 0
    while i < len(text):
        j = text.find("{", i)
        if j < 0:
            break
        try:
            obj, end = dec.raw_decode(text, j)
            objs.append((obj, j, end)); i = end
        except json.JSONDecodeError:
            i = j + 1
    objs = [o for o in objs if isinstance(o[0], dict)]
    if not objs:
        return bad("MALFORMED_JSON")
    if len(objs) > 1:
        return bad("MULTIPLE_ACTIONS")
    obj, s, e = objs[0]
    if text[:s].strip() or text[e:].strip():
        flags.append("extra_text")
    action = obj.get("action")
    if action not in VALID_ACTIONS:
        return bad("INVALID_ACTION")
    node = obj.get("node_id", None) if "node_id" in obj else None
    if "node_id" not in obj:
        flags.append("node_id_key_missing")
    if action == "STOP":
        if node is not None:
            return bad("CONFLICTING_STOP")
        return Decision("STOP", None, True, None, "STOP", "SLM_STOP", raw_output=raw, flags=flags)
    if node is None:
        return bad("MISSING_NODE_ID")
    if not isinstance(node, str):
        return bad("NODE_NOT_IN_CANDIDATES")
    node = node.strip()
    if node in alias_map["evidence"]:
        return bad("VISITED_NODE")
    if node not in alias_map["candidates"]:
        return bad("NODE_NOT_IN_CANDIDATES")
    return Decision("EXPLORE", alias_map["candidates"][node], True, None, "EXPLORE", None, alias=node,
                    raw_output=raw, flags=flags)


class TraversalController:
    def __init__(self, backend: Backend, cfg: Optional[ControllerConfig] = None):
        self.backend, self.cfg = backend, cfg or ControllerConfig()

    def choose_action(self, query, evidence: List[Passage], candidates: List[Passage], history: List[HistoryStep],
                      remaining_budget: int, previous_invalid: Optional[str] = None) -> Decision:
        """previous_invalid: raw text of an invalid reply to the SAME state; triggers the single correction retry
        (that reply is replayed as the assistant turn, followed by a fixed correction message)."""
        if not candidates:
            raise ValueError("choose_action requires a non-empty candidate set (empty -> NO_CANDIDATES upstream)")
        ids = [p.chunk_id for p in candidates]
        if len(set(ids)) != len(ids) or set(ids) & {p.chunk_id for p in evidence}:
            raise ValueError("candidates must be unique and disjoint from evidence")
        system, user, alias_map = build_prompt(query, evidence, candidates, history, remaining_budget, self.cfg)
        if previous_invalid is None:
            out = self.backend.generate(system, user, self.cfg.max_new_tokens, self.cfg.temperature)
        else:
            extra = [{"role": "assistant", "content": previous_invalid},
                     {"role": "user", "content": correction_message(len(candidates), self.cfg.prompt_version)}]
            out = self.backend.generate(system, user, self.cfg.max_new_tokens, self.cfg.temperature,
                                        extra_messages=extra)
        d = parse_action(out["text"], alias_map)
        d.prompt_tokens, d.output_tokens, d.latency_ms = out["prompt_tokens"], out["output_tokens"], out["latency_ms"]
        d.backend_stats = {k: v for k, v in out.items() if k not in ("text", "prompt_tokens", "output_tokens", "latency_ms")}
        return d

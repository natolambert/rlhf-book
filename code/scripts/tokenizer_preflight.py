# Pre-flight checklist for switching the SFT exercise to a new HF model.
#
# Tokenizer and chat-template behavior is NOT universal across model families
# (see https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior).
# This script probes the (base model, optional chat-template donor) pair the
# same way instruction_tuning/utils.py uses it, and prints a PASS/WARN/FAIL
# checklist of everything that could silently corrupt a prompt-masked SFT run.
#
# Usage (from code/):
#   uv run python scripts/tokenizer_preflight.py allenai/OLMo-2-0425-1B \
#       --chat-template-source allenai/OLMo-2-0425-1B-SFT
#   uv run python scripts/tokenizer_preflight.py Qwen/Qwen3-1.7B-Base \
#       --chat-template-source Qwen/Qwen3-1.7B
#
# Exit code: 0 = no FAIL, 1 = at least one FAIL.

import argparse
import re
from dataclasses import dataclass, field

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from transformers import __version__ as transformers_version


console = Console()

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"
STYLE = {PASS: "bold green", WARN: "bold yellow", FAIL: "bold red", INFO: "dim"}

# Conversations chosen to stress the prefix property and template branches.
TEST_CONVERSATIONS = {
    "single-turn": [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ],
    "multi-turn": [
        {"role": "user", "content": "Hi!"},
        {"role": "assistant", "content": "Hello, how can I help?"},
        {"role": "user", "content": "Name a prime number."},
        {"role": "assistant", "content": "7 is prime."},
    ],
    "with-system": [
        {"role": "system", "content": "You answer in one short sentence."},
        {"role": "user", "content": "What is water made of?"},
        {"role": "assistant", "content": "Water is H2O."},
    ],
    "leading-whitespace-answer": [
        {"role": "user", "content": "Continue: 1 2 3"},
        {"role": "assistant", "content": "    4 5 6"},
    ],
    "newline-start-answer": [
        {"role": "user", "content": "Write two lines."},
        {"role": "assistant", "content": "\nline one\nline two"},
    ],
    "unicode": [
        {"role": "user", "content": "日本の首都はどこですか？"},
        {"role": "assistant", "content": "東京です。"},
    ],
    # Reasoning templates (Qwen3-style) rewrite history: think blocks are kept in
    # some positions and stripped from others. Both _encode_row renders must agree.
    "think-in-history": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>\nEasy arithmetic.\n</think>\n\n2+2 is 4."},
        {"role": "user", "content": "And 3+3?"},
        {"role": "assistant", "content": "3+3 is 6."},
    ],
}

# Role-marker shapes used by the major template families.
MARKER_PATTERNS = [
    r"<\|[a-zA-Z_]+\|>",  # <|user|>, <|im_start|>, <|endoftext|> ...
    r"</?think>",  # Qwen3 / reasoning templates
    r"\[/?INST\]",  # Llama-2 style
    r"<<//?SYS>>",  # Llama-2 style
    r"<(?:start|end)_of_turn>",  # Gemma style
    r"<\|(?:begin|end|start)_of_text\|>",  # Llama-3 style
]


@dataclass
class Check:
    status: str
    name: str
    detail: str


@dataclass
class Report:
    sections: dict[str, list[Check]] = field(default_factory=dict)

    def add(self, section: str, status: str, name: str, detail: str = "") -> None:
        self.sections.setdefault(section, []).append(Check(status, name, detail))

    @property
    def n_fail(self) -> int:
        return sum(c.status == FAIL for cs in self.sections.values() for c in cs)


def render(tok, messages, generation_prompt=False, tokenize=True):
    """apply_chat_template with the exact flags the training code relies on."""
    return tok.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=generation_prompt,
        return_dict=False,
    )


def fmt_ids(tok, ids, limit=8):
    pieces = [repr(tok.decode([t])) for t in ids[:limit]]
    return " ".join(pieces) + (" ..." if len(ids) > limit else "")


# ---------------------------------------------------------------- sections


def check_loading(model_id, donor_id, rep):
    """Load tokenizer + config; lift the chat template like load_model() does."""
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
        rep.add("Loading", PASS, "Tokenizer loads with trust_remote_code=False")
    except Exception as e:  # noqa: BLE001 - report any loader failure the same way
        rep.add(
            "Loading",
            FAIL,
            "Tokenizer requires trust_remote_code or is broken",
            f"{type(e).__name__}: {e}. The training code passes trust_remote_code=False.",
        )
        return None, None, None
    rep.add(
        "Loading",
        INFO,
        "Tokenizer class",
        f"{type(tok).__name__} (fast={tok.is_fast}), transformers {transformers_version}",
    )

    donor = None
    if tok.chat_template is None:
        if donor_id is None:
            rep.add(
                "Loading",
                FAIL,
                "No chat template anywhere",
                "Base tokenizer has no chat_template and no --chat-template-source given.",
            )
            return tok, None, None
        donor = AutoTokenizer.from_pretrained(donor_id, trust_remote_code=False)
        if donor.chat_template is None:
            rep.add("Loading", FAIL, "Donor has no chat template", donor_id)
            return tok, donor, None
        tok.chat_template = donor.chat_template
        rep.add("Loading", PASS, "Chat template lifted from donor", donor_id)
    else:
        rep.add(
            "Loading",
            INFO if donor_id is None else WARN,
            "Base tokenizer already ships a chat template",
            "Donor ignored — remove chat_template_source or verify which template you want."
            if donor_id
            else "Model may be instruct-tuned already; the exercise expects a base model.",
        )
        if donor_id:
            donor = AutoTokenizer.from_pretrained(donor_id, trust_remote_code=False)

    cfg = None
    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    except Exception as e:  # noqa: BLE001
        rep.add("Loading", WARN, "AutoConfig failed (vocab/EOS cross-checks skipped)", str(e))
    return tok, donor, cfg


def check_special_tokens(tok, donor, cfg, rep):
    s = "Special tokens"
    for name in ("bos", "eos", "pad", "unk"):
        token = getattr(tok, f"{name}_token")
        tid = getattr(tok, f"{name}_token_id")
        rep.add(s, INFO, f"{name}_token", f"{token!r} (id={tid})")

    if tok.eos_token_id is None:
        rep.add(s, FAIL, "No EOS token", "Nothing can teach the model to stop.")
    if tok.bos_token_id is not None and tok.bos_token_id == tok.eos_token_id:
        rep.add(
            s,
            WARN,
            "BOS == EOS (OLMo-2-style overload)",
            "Same token opens and closes a conversation; meaning is positional. "
            "Fine for training, but never concatenate per-message renders.",
        )
    if tok.pad_token_id is None:
        rep.add(
            s,
            WARN,
            "No pad token (training code falls back to pad = eos)",
            "OK for this loop: labels at padding are -100 and attention_mask is built "
            "by the collator, so no EOS gradient leaks from padding.",
        )
    elif tok.pad_token_id == tok.eos_token_id:
        rep.add(
            s,
            WARN,
            "pad == eos",
            "Never mask labels with (input_ids == pad_token_id) — that erases the real "
            "EOS from the loss and the model never learns to stop. Mask by position "
            "like _collate does (this repo is safe).",
        )

    if cfg is not None:
        text_cfg = getattr(cfg, "text_config", None) or cfg
        vocab = getattr(text_cfg, "vocab_size", None)
        if vocab is not None:
            if len(tok) > vocab:
                rep.add(
                    s,
                    FAIL,
                    "len(tokenizer) > model vocab_size",
                    f"{len(tok)} > {vocab}: ids past the embedding table -> CUDA index "
                    "errors. Needs model.resize_token_embeddings(len(tokenizer)).",
                )
            elif vocab > len(tok):
                rep.add(
                    s,
                    INFO,
                    "model vocab_size > len(tokenizer)",
                    f"{vocab} > {len(tok)}: padded embedding rows, harmless.",
                )
            else:
                rep.add(s, PASS, "len(tokenizer) == model vocab_size", str(vocab))
        cfg_eos = getattr(text_cfg, "eos_token_id", None)
        if cfg_eos is not None:
            cfg_eos_list = cfg_eos if isinstance(cfg_eos, list) else [cfg_eos]
            if tok.eos_token_id not in cfg_eos_list:
                rep.add(
                    s,
                    WARN,
                    "config.eos_token_id != tokenizer.eos_token_id",
                    f"config {cfg_eos} vs tokenizer {tok.eos_token_id}; generate() stop "
                    "behavior may not match training labels.",
                )

    if tok.model_max_length and tok.model_max_length > 1_000_000:
        rep.add(
            s,
            WARN,
            "model_max_length is a placeholder",
            f"{tok.model_max_length:.0e} — don't rely on it for truncation; "
            "set max_length explicitly in the config (the exercise does).",
        )
    if cfg is not None:
        mpe = getattr(getattr(cfg, "text_config", None) or cfg, "max_position_embeddings", None)
        if mpe is not None:
            rep.add(
                s,
                INFO,
                "max_position_embeddings",
                f"{mpe} — keep the training config's max_length at or below this.",
            )

    if donor is not None and donor.eos_token != tok.eos_token:
        rep.add(
            s,
            WARN,
            "Donor EOS differs from base EOS",
            f"base {tok.eos_token!r} vs donor {donor.eos_token!r}. The lifted template "
            "may close turns with the donor's literal token while generate() stops on "
            "the base EOS — see the 'training EOS' check below.",
        )


def check_bare_tokenizer(tok, rep):
    s = "Bare tokenizer"
    text = "Hello world"
    default = tok(text)["input_ids"]
    bare = tok(text, add_special_tokens=False)["input_ids"]
    added_front = default[: len(default) - len(bare)] if default[-len(bare) :] == bare else []
    added_back = default[len(bare) :] if default[: len(bare)] == bare else []
    if default == bare:
        rep.add(s, INFO, "tokenizer() adds no special tokens by default", "")
    else:
        where = []
        if added_front:
            where.append(f"prepends {fmt_ids(tok, added_front)}")
        if added_back:
            where.append(f"appends {fmt_ids(tok, added_back)}")
        rep.add(
            s,
            WARN,
            "tokenizer() auto-adds special tokens",
            f"{' and '.join(where) or 'changes ids'} — tokenizing template-rendered text "
            "with default flags duplicates them (see 'double-BOS' check).",
        )

    for probe in ("Hello world", " leading space", "東京です。", "tabs\tand\nnewlines"):
        ids = tok(probe, add_special_tokens=False)["input_ids"]
        if tok.decode(ids) != probe:
            rep.add(
                s,
                WARN,
                "encode->decode does not round-trip",
                f"{probe!r} -> {tok.decode(ids)!r} (decode artifacts; affects log/diff "
                "tooling, not training itself).",
            )
            break
    else:
        rep.add(s, PASS, "encode->decode round-trips on probe strings")

    a, b = "answer:\n", "    indented"
    if tok(a + b, add_special_tokens=False)["input_ids"] != (
        tok(a, add_special_tokens=False)["input_ids"]
        + tok(b, add_special_tokens=False)["input_ids"]
    ):
        rep.add(
            s,
            INFO,
            "BPE is not concatenation-safe at boundaries (expected)",
            "tokenize(A)+tokenize(B) != tokenize(A+B) — why _encode_row renders the "
            "full conversation instead of stitching per-message ids.",
        )


def check_chat_template(tok, rep):
    s = "Chat template"
    msgs = TEST_CONVERSATIONS["single-turn"]

    text = render(tok, msgs, tokenize=False)
    rep.add(s, INFO, "Rendered single-turn conversation", repr(text))

    # Boilerplate volume: a big remainder usually means an injected default system
    # prompt (Qwen2.5-style) the training data would silently inherit.
    leftover = text
    for m in msgs:
        leftover = leftover.replace(m["content"], "")
    if len(leftover) > 120:
        rep.add(
            s,
            WARN,
            "Template injects a lot of boilerplate",
            f"{len(leftover)} chars beyond message contents — check for a default "
            f"system prompt: {leftover!r}",
        )

    prompt_text = render(tok, msgs[:-1], generation_prompt=True, tokenize=False)
    no_gen = render(tok, msgs[:-1], generation_prompt=False, tokenize=False)
    header = prompt_text[len(no_gen) :] if prompt_text.startswith(no_gen) else None
    if prompt_text == no_gen:
        rep.add(
            s,
            FAIL,
            "add_generation_prompt does nothing",
            "Without an assistant header the prefix trick masks nothing of the header "
            "and inference prompts are malformed.",
        )
    elif header is not None:
        rep.add(s, PASS, "Generation prompt appends the assistant header", repr(header))
    else:
        rep.add(
            s,
            WARN,
            "add_generation_prompt rewrites earlier text",
            "Header is not a pure suffix; inspect the template.",
        )

    if re.search(r"<think>|\{%-?\s*if\s+[^%]*thinking", tok.chat_template or ""):
        rep.add(
            s,
            WARN,
            "Template contains thinking/<think> handling",
            "Reasoning-style template: rendered turns or generation prompts may include "
            "<think> blocks that end up supervised or change the stop convention.",
        )
    if "generation" in (tok.chat_template or "") and "{% generation" in tok.chat_template:
        rep.add(
            s,
            INFO,
            "Template supports {% generation %}",
            "return_assistant_tokens_mask=True is available as an alternative to the "
            "prefix trick (masks every assistant turn, not just the last).",
        )

    for marker in sorted({m for p in MARKER_PATTERNS for m in re.findall(p, text + prompt_text)}):
        ids = tok(marker, add_special_tokens=False)["input_ids"]
        atomic = len(ids) == 1
        rep.add(
            s,
            INFO,
            f"Role marker {marker!r}",
            "single special token"
            if atomic
            else f"{len(ids)} plain BPE pieces (base model sees ordinary text, "
            "like OLMo-2's <|user|>)",
        )

    try:
        with_flag = tok.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        if with_flag != prompt_text:
            rep.add(
                s,
                WARN,
                "Template has enable_thinking modes (Qwen3-style)",
                f"enable_thinking=False changes the generation prompt to "
                f"{with_flag[len(no_gen) :]!r} — pick ONE mode and use it consistently "
                "for training and sampling.",
            )
    except Exception:  # noqa: BLE001 - template doesn't take the kwarg; nothing to flag
        pass

    for probe_name, probe in [
        ("assistant-first", [{"role": "assistant", "content": "hi"}]),
        ("consecutive-user", [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]),
    ]:
        try:
            render(tok, probe, tokenize=False)
        except Exception as e:  # noqa: BLE001
            rep.add(
                s,
                INFO,
                f"Template rejects {probe_name} conversations",
                f"{type(e).__name__}: fine for no_robots; matters for odd datasets.",
            )


def check_prefix_property(tok, rep):
    """The invariant _encode_row depends on, across all test conversations."""
    s = "Prefix property (_encode_row invariant)"
    for name, msgs in TEST_CONVERSATIONS.items():
        try:
            prompt_ids = render(tok, msgs[:-1], generation_prompt=True)
            full_ids = render(tok, msgs)
        except Exception as e:  # noqa: BLE001
            rep.add(
                s,
                WARN if name == "with-system" else FAIL,
                f"[{name}] template raised",
                f"{type(e).__name__}: {e}"
                + (" — drop system turns for this family." if name == "with-system" else ""),
            )
            continue
        if full_ids[: len(prompt_ids)] == prompt_ids:
            suffix = full_ids[len(prompt_ids) :]
            rep.add(
                s,
                PASS,
                f"[{name}] prompt is an exact token prefix",
                f"supervised suffix: {fmt_ids(tok, suffix, limit=12)}",
            )
        else:
            n = next(
                (i for i, (x, y) in enumerate(zip(prompt_ids, full_ids, strict=False)) if x != y),
                min(len(prompt_ids), len(full_ids)),
            )
            rep.add(
                s,
                FAIL,
                f"[{name}] PREFIX PROPERTY VIOLATED at token {n}",
                f"prompt ..{fmt_ids(tok, prompt_ids[max(0, n - 2) : n + 3])} vs "
                f"full ..{fmt_ids(tok, full_ids[max(0, n - 2) : n + 3])} — "
                "_encode_row would mask the wrong tokens for this conversation shape.",
            )


def check_training_eos(tok, cfg, model_id, rep):
    s = "Training EOS / stop convention"
    msgs = TEST_CONVERSATIONS["single-turn"]
    full_ids = render(tok, msgs)

    # The token the template actually closes the assistant turn with: last token,
    # skipping a trailing whitespace token (Qwen/SmolLM templates end '<|im_end|>\n').
    closing_id = full_ids[-1]
    trailing = []
    if tok.decode([closing_id]).strip() == "" and len(full_ids) > 1:
        trailing = [full_ids[-1]]
        closing_id = full_ids[-2]
    closing = tok.decode([closing_id])

    # transformers generate() stops on model.generation_config.eos_token_id — NOT on
    # tokenizer.eos_token (QwenLM/Qwen3#927). vLLM, by contrast, uses the tokenizer's.
    gen_eos = None
    try:
        gen_eos = GenerationConfig.from_pretrained(model_id).eos_token_id
    except Exception:  # noqa: BLE001 - no generation_config.json; generate() falls back
        if cfg is not None:
            gen_eos = getattr(getattr(cfg, "text_config", None) or cfg, "eos_token_id", None)
    gen_eos_list = gen_eos if isinstance(gen_eos, list) else ([] if gen_eos is None else [gen_eos])
    rep.add(
        s,
        INFO,
        "Stop tokens by consumer",
        f"template closes turns with {closing!r} (id={closing_id}); "
        f"HF generate() stops on generation_config {gen_eos_list} "
        f"({[tok.decode([i]) for i in gen_eos_list]}); "
        f"vLLM stops on tokenizer eos {tok.eos_token!r} (id={tok.eos_token_id}).",
    )

    if closing_id in gen_eos_list and closing_id == tok.eos_token_id:
        rep.add(
            s,
            PASS,
            "Template stop token == generation_config EOS == tokenizer EOS",
            f"{closing!r} is the last supervised label and every stack stops on it.",
        )
    elif closing_id in gen_eos_list:
        rep.add(
            s,
            WARN,
            "HF generate() stops, but tokenizer EOS differs from the template's stop token",
            f"generation_config covers {closing!r}, so this repo's sampling works; but "
            f"vLLM and anything reading tokenizer.eos_token ({tok.eos_token!r}) won't "
            f"stop. Set tokenizer.eos_token = {closing!r} before serving elsewhere.",
        )
    else:
        rep.add(
            s,
            FAIL,
            "Template's stop token is not in generation_config eos_token_id",
            f"model learns to emit {closing!r} (id={closing_id}) but generate() stops "
            f"only on {gen_eos_list} -> endless generations. Official fix (QwenLM/"
            f"Qwen3#927, what Qwen's own instruct models ship): "
            f"model.generation_config.eos_token_id = {sorted(set(gen_eos_list + [closing_id]))} "
            f"after loading. Note: setting tokenizer.eos_token alone does NOT fix HF "
            f"generate(); set it too ({closing!r}) for vLLM/third-party stacks.",
        )
    if trailing:
        rep.add(
            s,
            INFO,
            "Token after the stop token in the render",
            f"{fmt_ids(tok, trailing)} also gets a supervised label — it can never be "
            "produced at inference (generation stops first); harmless.",
        )

    multi = TEST_CONVERSATIONS["multi-turn"]
    try:
        multi_text = render(tok, multi, tokenize=False)
        n_assistant = sum(m["role"] == "assistant" for m in multi)
        closes = multi_text.count(tok.eos_token) if tok.eos_token else 0
        if 0 < closes < n_assistant:
            rep.add(
                s,
                INFO,
                "EOS closes only some assistant turns",
                f"{closes} EOS for {n_assistant} assistant turns — irrelevant while "
                "training only the final turn, matters for train-on-all-turns recipes.",
            )
    except Exception:  # noqa: BLE001 - already reported by the prefix check
        pass


def check_double_bos(tok, rep):
    s = "Inference-path consistency"
    msgs = TEST_CONVERSATIONS["single-turn"]
    prompt_text = render(tok, msgs[:-1], generation_prompt=True, tokenize=False)
    via_template = render(tok, msgs[:-1], generation_prompt=True)
    via_text = tok(prompt_text)["input_ids"]  # generate_samples() tokenizes like this
    if via_template == via_text:
        rep.add(
            s,
            PASS,
            "tokenizer(rendered_text) == apply_chat_template ids",
            "generate_samples()-style tokenization matches training exactly.",
        )
    else:
        rep.add(
            s,
            FAIL,
            "tokenizer(rendered_text) != apply_chat_template ids (double-BOS?)",
            f"text-path starts {fmt_ids(tok, via_text)} vs template-path "
            f"{fmt_ids(tok, via_template)} — pass add_special_tokens=False when "
            "tokenizing rendered text (Llama-style tokenizers auto-prepend BOS on top "
            "of the template's).",
        )

    probe = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
    if not isinstance(probe, list):
        rep.add(
            s,
            INFO,
            "apply_chat_template(tokenize=True) returns a dict by default",
            f"{type(probe).__name__} in transformers {transformers_version} — keep "
            "passing return_dict=False like the training code does.",
        )


# ---------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="HF repo id of the (base) model to fine-tune")
    parser.add_argument(
        "--chat-template-source",
        default=None,
        help="Instruct sibling to lift the chat template from (config: chat_template_source)",
    )
    args = parser.parse_args()

    rep = Report()
    tok, donor, cfg = check_loading(args.model, args.chat_template_source, rep)
    if tok is not None and tok.chat_template is not None:
        check_special_tokens(tok, donor, cfg, rep)
        check_bare_tokenizer(tok, rep)
        check_chat_template(tok, rep)
        check_prefix_property(tok, rep)
        check_training_eos(tok, cfg, args.model, rep)
        check_double_bos(tok, rep)

    console.print(
        Panel(
            f"[bold]{args.model}[/bold]"
            + (
                f"  +  template from [bold]{args.chat_template_source}[/bold]"
                if args.chat_template_source
                else ""
            ),
            title="[bold cyan]Tokenizer pre-flight checklist[/bold cyan]",
            border_style="cyan",
        )
    )
    for section, checks in rep.sections.items():
        lines = []
        for c in checks:
            lines.append(
                f"[{STYLE[c.status]}]{c.status:>4}[/{STYLE[c.status]}]  [bold]{escape(c.name)}[/bold]"
                + (f"\n      [dim]{escape(c.detail)}[/dim]" if c.detail else "")
            )
        console.print(Panel("\n".join(lines), title=section, border_style="blue"))

    n_fail = rep.n_fail
    n_warn = sum(c.status == WARN for cs in rep.sections.values() for c in cs)
    style = "bold red" if n_fail else ("bold yellow" if n_warn else "bold green")
    console.print(f"[{style}]{n_fail} FAIL, {n_warn} WARN[/{style}]")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

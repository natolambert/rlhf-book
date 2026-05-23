import ast
import base64
import io
import json
import multiprocessing
import pickle
import re
import sys
import zlib
from datetime import datetime

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


CODE_PROMPT = (
    "You are a coding expert. You will be given a coding problem, and you need to "
    "write a correct Python program that matches the specification and passes all "
    "tests. The time limit is 1 second. You may start by outlining your thought "
    "process. In the end, please provide the complete code in a code block enclosed "
    "with ``` ```.\n\n{problem}"
)

LCB_DATE_CUTOFF = datetime(2025, 2, 1)  # split: before = train, on/after = eval
LCB_UNTIL = datetime(2025, 5, 1)
TIME_LIMIT = 6

INCORRECT_FORMAT = "Incorrect format"
TIMEOUT = "Time out"
ERROR_PREFIX = "Error: "


def _parse_signature(starter_code: str) -> str:
    return "def " + starter_code.split("def ")[1].split("Input\n")[0].strip()


def _decode_tests(encoded: str, fn_name: str) -> dict:
    tests = json.loads(pickle.loads(zlib.decompress(base64.b64decode(encoded))))
    return {
        "inputs": [t["input"] for t in tests],
        "outputs": [t["output"] for t in tests],
        "testtype": tests[0]["testtype"],  # "functional" (call a fn) or "stdin" (run a script)
        "fn_name": fn_name,
        "time_limit": TIME_LIMIT,
    }


def load_livecodebench(split: str) -> list[dict]:
    ds = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        revision="refs/pr/6",
        trust_remote_code=True,
    )
    if split == "train":
        ds = ds.filter(lambda ex: ex["contest_date"] < LCB_DATE_CUTOFF)
    else:
        ds = ds.filter(lambda ex: LCB_DATE_CUTOFF <= ex["contest_date"] < LCB_UNTIL)

    samples = []
    for i, ex in enumerate(ds):
        problem = ex["question_content"]
        if ex["starter_code"].strip():
            sig = _parse_signature(ex["starter_code"]).replace("(self, ", "(")
            problem += (
                f"\n\nYour solution should have the following signature: ```python\n{sig}\n```"
            )
        meta = json.loads(ex["metadata"]) if ex["metadata"].strip() else {}
        samples.append(
            {
                "id": str(i),
                "prompt": CODE_PROMPT.format(problem=problem),
                "tests": _decode_tests(ex["private_test_cases"], meta.get("func_name", "")),
            }
        )
    return samples


class LiveCodeBenchDataset(Dataset):
    def __init__(self, split: str):
        self.samples = load_livecodebench(split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def build_dataloader(split: str, batch_size: int = 8, shuffle: bool | None = None) -> DataLoader:
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        LiveCodeBenchDataset(split),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=list,
    )


def extract_code(response: str) -> str | None:
    blocks = re.findall(r"```(?:\w*)\n(.*?)```", response, re.DOTALL)
    return max(blocks, key=len) if blocks else None


def _reliability_guard() -> None:
    """Neutralize the most destructive syscalls before running model code.

    NOTE: This is a guard against accidents, NOT a security sandbox.
    """
    import builtins
    import os
    import shutil

    builtins.exit = builtins.quit = builtins.open = None
    for name in (
        "system",
        "remove",
        "removedirs",
        "rmdir",
        "kill",
        "unlink",
        "rename",
        "replace",
        "chmod",
        "chown",
    ):
        setattr(os, name, None)
    shutil.rmtree = shutil.move = None


def _infer_fn(code: str, fn_name: str) -> str:
    try:
        defs = [n.name for n in ast.parse(code).body if isinstance(n, ast.FunctionDef)]
    except SyntaxError:
        return fn_name
    if fn_name in defs:
        return fn_name
    return defs[0] if defs else fn_name


def _run_functional(code: str, fn_name: str, test_input: str, test_output: str):
    ns: dict = {"__name__": "__not_main__"}
    exec(code, ns)
    fn = ns.get(_infer_fn(code, fn_name))
    if not callable(fn):
        raise NameError(f"function {fn_name!r} is not defined")
    args = [json.loads(line) for line in test_input.split("\n") if line.strip()]
    result = fn(*args)
    expected = json.loads(test_output)
    try:
        ok = json.dumps(result, sort_keys=True) == json.dumps(expected, sort_keys=True)
    except TypeError:
        ok = result == expected
    return ok, result


def _run_stdin(code: str, test_input: str, test_output: str, stdout: io.StringIO):
    old_stdin = sys.stdin
    sys.stdin = io.StringIO(test_input)
    try:
        exec(compile('__name__ = "__main__"\n' + code, "solution.py", "exec"), {})
        produced = stdout.getvalue().strip().replace("\r", "").replace("\n", " ")
        expected = test_output.strip().replace("\r", "").replace("\n", " ")
        return produced == expected, stdout.getvalue().strip()
    finally:
        sys.stdin = old_stdin


def _worker(
    conn, testtype: str, code: str, fn_name: str, test_input: str, test_output: str
) -> None:
    _reliability_guard()
    stdout = io.StringIO()
    sys.stdout = stdout
    try:
        if testtype == "functional":
            passed, actual = _run_functional(code, fn_name, test_input, test_output)
        else:
            passed, actual = _run_stdin(code, test_input, test_output, stdout)
    except BaseException as e:
        passed, actual = False, f"{ERROR_PREFIX}{type(e).__name__}: {e}"
    finally:
        sys.stdout = sys.__stdout__
    if not isinstance(actual, (str, int, float, bool, type(None))):
        actual = repr(actual)
    conn.send({"passed": passed, "actual": actual})
    conn.close()


def _run_one(
    testtype: str, code: str, fn_name: str, test_input: str, test_output: str, timeout: float
) -> dict:
    ctx = multiprocessing.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_worker, args=(child, testtype, code, fn_name, test_input, test_output))
    p.start()
    child.close()
    record = {"passed": False, "actual": TIMEOUT, "input": test_input, "expected": test_output}
    if parent.poll(timeout):
        try:
            record.update(parent.recv())
        except EOFError:
            record["actual"] = f"{ERROR_PREFIX}process crashed"
    if p.is_alive():
        p.kill()
    p.join()
    parent.close()
    return record


def run_tests(tests: dict, completion: str, max_tests: int | None = None) -> list[dict]:
    n = len(tests["inputs"])
    if max_tests:
        n = min(n, max_tests)
    timeout = float(tests.get("time_limit") or TIME_LIMIT)
    records = []
    for i in range(n):
        record = _run_one(
            tests["testtype"],
            completion,
            tests.get("fn_name", ""),
            tests["inputs"][i],
            tests["outputs"][i],
            timeout,
        )
        record["test_idx"] = i
        records.append(record)
    return records


def _clip(value, limit: int = 250) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "..."


def format_feedback(records: list[dict], max_chars: int = 2000) -> str:
    failing = [r for r in records if not r["passed"]]
    if not failing:
        return ""

    error = next((r for r in failing if str(r["actual"]).startswith(ERROR_PREFIX)), None)
    timeout = next((r for r in failing if r["actual"] == TIMEOUT), None)
    r = error or timeout or min(failing, key=lambda x: len(str(x["input"])) + len(str(x["actual"])))
    actual = r["actual"]

    if actual == INCORRECT_FORMAT:
        feedback = "Incorrect Format: put your final code inside a ```python ... ``` block."
    elif actual == TIMEOUT:
        feedback = f"Time Limit Exceeded\n\nInput:\n{_clip(r['input'])}"
    elif str(actual).startswith(ERROR_PREFIX):
        feedback = f"Runtime Error\n{actual[len(ERROR_PREFIX) :]}\n\nInput:\n{_clip(r['input'])}"
    else:
        feedback = "\n".join(
            [
                f"Wrong Answer on test {r['test_idx'] + 1}\n",
                f"Input:\n{_clip(r['input'])}",
                f"Output:\n{_clip(actual)}",
                f"Expected:\n{_clip(r['expected'])}",
            ]
        )
    return feedback[:max_chars]


def compute_score(response: str, tests: dict, max_tests: int | None = None) -> dict:
    completion = extract_code(response)
    if completion is None:
        records = [
            {
                "test_idx": 0,
                "passed": False,
                "actual": INCORRECT_FORMAT,
                "input": None,
                "expected": None,
            }
        ]
    else:
        records = run_tests(tests, completion, max_tests=max_tests)

    acc = sum(r["passed"] for r in records) / len(records)
    return {
        "reward": 1.0 if acc == 1.0 else 0.0,
        "acc": acc,
        "feedback": format_feedback(records),
        "num_tests": len(records),
    }

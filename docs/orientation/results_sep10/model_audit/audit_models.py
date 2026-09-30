"""Audit every checkpoint the paper uses or will use (ORCH_DIRECTIVE_2).

For one model per invocation (so only one is in memory), record:
  - repo id, revision SHA, quantization, model_type, converter version
  - loader path the harness actually takes (mlx_lm, or the mlx_vlm fallback)
  - tokenizer class, BOS/EOS, what `add_special_tokens=True` adds, and the BOS
    id the matched-set scorer prepends (same code path as score_matched_regime)
  - chat template presence (the matched set is scored as raw text, no template)
  - sanity checks:
      known-answer pairs  36 true/false claims in the matched set's frame
                          ("The ROLE of ENTITY is PERSON."), scored with the
                          scorer's own `claim_logprob`; pairwise accuracy on
                          raw mean log-prob and on PMI (the scorer's metric)
      BOS ablation        the same pairs without the BOS prefix
      fluency             mean NLL per token on a plain English paragraph
      generation          5 capital-city questions, greedy, through the chat
                          template (informational: the matched set does not
                          use generation)

Usage (from repo root):
    python3 docs/orientation/results_sep10/model_audit/audit_models.py --base <path-or-repo> --name <tag>
    python3 docs/orientation/results_sep10/model_audit/audit_models.py --adapter <dir> --name tsct
    add --force-vlm to load through mlx_vlm instead of mlx_lm
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.training.generate_predictions import (  # noqa: E402
    _hf_tok, _load_multimodal, _logits, load_base_only, rebuild_adapter)
from src.training.score_matched_regime import claim_logprob  # noqa: E402

OUT = Path(__file__).resolve().parent / "runs"

# (role, entity, true person, false person)
KNOWN = [
    ("founder", "Microsoft", "Bill Gates", "Larry Page"),
    ("founder", "Apple", "Steve Jobs", "Jeff Bezos"),
    ("founder", "Amazon", "Jeff Bezos", "Bill Gates"),
    ("founder", "Facebook", "Mark Zuckerberg", "Steve Jobs"),
    ("founder", "Google", "Larry Page", "Mark Zuckerberg"),
    ("founder", "the Ford Motor Company", "Henry Ford", "Walt Disney"),
    ("founder", "the Walt Disney Company", "Walt Disney", "Henry Ford"),
    ("founder", "SpaceX", "Elon Musk", "Richard Branson"),
    ("founder", "Virgin Group", "Richard Branson", "Elon Musk"),
    ("founder", "Oracle", "Larry Ellison", "Michael Dell"),
    ("founder", "Dell", "Michael Dell", "Larry Ellison"),
    ("founder", "Nike", "Phil Knight", "Ray Kroc"),
    ("founder", "the Red Cross", "Henry Dunant", "Florence Nightingale"),
    ("founder", "Wikipedia", "Jimmy Wales", "Tim Berners-Lee"),
    ("author", "Hamlet", "William Shakespeare", "Charles Dickens"),
    ("author", "Pride and Prejudice", "Jane Austen", "Mark Twain"),
    ("author", "Nineteen Eighty-Four", "George Orwell", "Jane Austen"),
    ("author", "War and Peace", "Leo Tolstoy", "Victor Hugo"),
    ("author", "Les Misérables", "Victor Hugo", "Leo Tolstoy"),
    ("author", "On the Origin of Species", "Charles Darwin", "Isaac Newton"),
    ("author", "Don Quixote", "Miguel de Cervantes", "Dante Alighieri"),
    ("author", "the Divine Comedy", "Dante Alighieri", "Miguel de Cervantes"),
    ("author", "Oliver Twist", "Charles Dickens", "William Shakespeare"),
    ("author", "The Adventures of Tom Sawyer", "Mark Twain", "George Orwell"),
    ("composer", "the Moonlight Sonata", "Ludwig van Beethoven", "Johann Sebastian Bach"),
    ("composer", "The Magic Flute", "Wolfgang Amadeus Mozart", "Giuseppe Verdi"),
    ("composer", "the Brandenburg Concertos", "Johann Sebastian Bach", "Frédéric Chopin"),
    ("composer", "Aida", "Giuseppe Verdi", "Wolfgang Amadeus Mozart"),
    ("painter", "the Mona Lisa", "Leonardo da Vinci", "Vincent van Gogh"),
    ("painter", "The Starry Night", "Vincent van Gogh", "Pablo Picasso"),
    ("painter", "Guernica", "Pablo Picasso", "Claude Monet"),
    ("painter", "The Last Supper", "Leonardo da Vinci", "Rembrandt"),
    ("discoverer", "penicillin", "Alexander Fleming", "Louis Pasteur"),
    ("discoverer", "the theory of general relativity", "Albert Einstein", "Niels Bohr"),
    ("inventor", "the telephone", "Alexander Graham Bell", "Thomas Edison"),
    ("inventor", "the World Wide Web", "Tim Berners-Lee", "Jimmy Wales"),
]

PARAGRAPH = (
    "The river rises in the hills to the north and flows south through farmland "
    "before reaching the sea. In spring the water is high and fast, and the old "
    "bridge in the town is sometimes closed. Most of the people who live along "
    "its banks work on farms or in the small factories near the harbour."
)

QUESTIONS = [("What is the capital of France?", "paris"),
             ("What is the capital of Japan?", "tokyo"),
             ("What is the capital of Italy?", "rome"),
             ("What is the capital of Egypt?", "cairo"),
             ("What is the capital of Canada?", "ottawa")]


def repo_meta(path):
    p = Path(path)
    meta = {"path": str(p)}
    if not p.exists():
        return meta
    readme = p / "README.md"
    if readme.exists():
        t = readme.read_text()
        m = re.search(r"^# (\S+/\S+)", t, re.M)
        meta["repo_id"] = m.group(1) if m else None
        m = re.search(r"base_model:\s*(\S+)", t)
        meta["base_model"] = m.group(1) if m else None
        m = re.search(r"(mlx-(?:vlm|lm)) version \*\*([\d.]+)\*\*", t)
        meta["converter"] = f"{m.group(1)} {m.group(2)}" if m else None
    md = sorted((p / ".cache/huggingface/download").glob("*.metadata"))
    meta["revision_sha"] = md[0].read_text().splitlines()[0] if md else None
    cfg = json.load(open(p / "config.json"))
    q = cfg.get("quantization") or cfg.get("quantization_config") or {}
    meta["quantization"] = {k: q.get(k) for k in ("bits", "group_size", "mode") if k in q}
    meta["model_type"] = cfg.get("model_type")
    meta["chat_template_file"] = any((p / f).exists() for f in ("chat_template.jinja", "chat_template.json"))
    tc = p / "tokenizer_config.json"
    if tc.exists():
        tcfg = json.load(open(tc))
        meta["tokenizer_class"] = tcfg.get("tokenizer_class")
        meta["tokenizer_config_bos"] = tcfg.get("bos_token")
        meta["add_bos_token"] = tcfg.get("add_bos_token")
    return meta


def hf_cache_meta(repo):
    """Revision SHA for a repo loaded from the HF hub cache."""
    from huggingface_hub import constants
    d = Path(constants.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))
    ref = d / "refs" / "main"
    snap = None
    if ref.exists():
        snap = ref.read_text().strip()
    out = {"repo_id": repo, "revision_sha": snap}
    if snap:
        out.update({k: v for k, v in repo_meta(d / "snapshots" / snap).items() if k not in ("path", "revision_sha")})
    return out


def pairwise(model, hf_tok, bos_id):
    rows, wins_raw, wins_pmi = [], 0, 0
    name_cache = {}

    def lp(t):
        return claim_logprob(model, hf_tok, t, bos_id)

    for role, ent, tp, fp in KNOWN:
        t_txt, f_txt = f"The {role} of {ent} is {tp}.", f"The {role} of {ent} is {fp}."
        a, b = lp(t_txt), lp(f_txt)
        for n in (tp, fp):
            if n not in name_cache:
                name_cache[n] = lp(f"{n} is a person.")
        pa, pb = a - name_cache[tp], b - name_cache[fp]
        wins_raw += a > b
        wins_pmi += pa > pb
        rows.append({"true": t_txt, "raw_true": a, "raw_false": b, "pmi_true": pa, "pmi_false": pb})
    n = len(KNOWN)
    return {"acc_raw": wins_raw / n, "acc_pmi": wins_pmi / n, "n": n, "rows": rows}


def nll(model, hf_tok, bos_id):
    return -claim_logprob(model, hf_tok, PARAGRAPH, bos_id)


def generate(model, hf_tok, max_new=48):
    out = []
    for q, ans in QUESTIONS:
        msgs = [{"role": "user", "content": q + " Answer in one word."}]
        try:
            prompt = hf_tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                                enable_thinking=False)
        except Exception as exc:  # no template
            out.append({"q": q, "error": f"{type(exc).__name__}: {exc}"})
            continue
        # the rendered template already carries any BOS the model expects
        ids = hf_tok(prompt, add_special_tokens=False)["input_ids"]
        eos = {hf_tok.eos_token_id} if hf_tok.eos_token_id is not None else set()
        gen = []
        for _ in range(max_new):
            logits = _logits(model(mx.array([ids + gen])))[0, -1]
            nxt = int(mx.argmax(logits).item())
            if nxt in eos:
                break
            gen.append(nxt)
        text = hf_tok.decode(gen)
        out.append({"q": q, "text": text, "correct": ans in text.lower()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument("--adapter")
    ap.add_argument("--name", required=True)
    ap.add_argument("--force-vlm", action="store_true")
    ap.add_argument("--no-generate", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rec = {"name": args.name, "force_vlm": args.force_vlm}

    if args.adapter:
        cfg = json.load(open(Path(args.adapter) / "adapter_config.json"))
        rec["meta"] = hf_cache_meta(cfg["model"]) | {"adapter": args.adapter}
        model, tokenizer, _, _ = rebuild_adapter(args.adapter)
        rec["loader"] = "mlx_lm (rebuild_adapter)"
    else:
        rec["meta"] = repo_meta(args.base) if Path(args.base).exists() else hf_cache_meta(args.base)
        if args.force_vlm:
            model, tokenizer = _load_multimodal(args.base)
            rec["loader"] = "mlx_vlm (forced)"
        else:
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                model, tokenizer, _, info = load_base_only(args.base)
            rec["loader"] = "mlx_vlm (fallback)" if "retrying via mlx_vlm" in buf.getvalue() else "mlx_lm"
    rec["load_seconds"] = round(time.time() - t0, 1)

    hf_tok = _hf_tok(tokenizer)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is None:
        bos_id = getattr(hf_tok, "bos_token_id", None)
    probe = "The founder of Nike is Phil Knight."
    with_sp = hf_tok(probe, add_special_tokens=True)["input_ids"]
    without = hf_tok(probe, add_special_tokens=False)["input_ids"]
    rec["tokenizer"] = {
        "class": type(hf_tok).__name__, "len": len(hf_tok) if hasattr(hf_tok, "__len__") else None,
        "bos_token": getattr(hf_tok, "bos_token", None), "bos_id_scorer_uses": bos_id,
        "eos_token": getattr(hf_tok, "eos_token", None),
        "add_special_tokens_adds": [t for t in with_sp if t not in without][:4],
        "first_ids_scored": ([bos_id] if bos_id is not None and with_sp[:1] != [bos_id] else []) + with_sp[:6],
        "has_chat_template": bool(getattr(hf_tok, "chat_template", None)),
    }

    t1 = time.time()
    pw = pairwise(model, hf_tok, bos_id)
    rec["known_answer"] = {k: v for k, v in pw.items() if k != "rows"}
    rec["known_answer_rows"] = pw["rows"]
    if bos_id is not None:
        nb = pairwise(model, hf_tok, None)
        rec["known_answer_no_bos"] = {k: v for k, v in nb.items() if k != "rows"}
        rec["nll_no_bos"] = nll(model, hf_tok, None)
    rec["nll_paragraph"] = nll(model, hf_tok, bos_id)
    if not args.no_generate:
        rec["generation"] = generate(model, hf_tok)
    rec["score_seconds"] = round(time.time() - t1, 1)
    tag = args.name + ("_vlm" if args.force_vlm else "")
    json.dump(rec, open(OUT / f"{tag}.json", "w"), indent=2, default=str)
    s = {k: rec[k] for k in ("name", "loader") if k in rec}
    s |= {"acc_raw": pw["acc_raw"], "acc_pmi": pw["acc_pmi"], "nll": round(rec["nll_paragraph"], 3),
          "gen_correct": sum(g.get("correct", False) for g in rec.get("generation", [])),
          "bos": bos_id, "no_bos": rec.get("known_answer_no_bos"), "secs": round(time.time() - t0)}
    print(json.dumps(s))


if __name__ == "__main__":
    main()

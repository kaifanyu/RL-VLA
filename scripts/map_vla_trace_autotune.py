"""Offline, audited autotune-config mapping; never edits a shared cache or runs a model."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def proto_fields(text):
    """Parse message boundaries while preserving source spans and quoted HLO text."""
    pattern = re.compile(r'\s+|\#[^\n]*|//[^\n]*|"(?:\\.|[^"\\])*"|[{}<>:]|[^\s{}<>:]+')
    tokens = [(match.group(), match.start(), match.end()) for match in pattern.finditer(text)
              if not match.group().isspace() and not match.group().startswith(("#", "//"))]
    cursor = 0

    def message(closing=None):
        nonlocal cursor
        fields = []
        while cursor < len(tokens):
            name, start, _ = tokens[cursor]
            if name == closing:
                cursor += 1
                return fields, tokens[cursor - 1][2]
            if name in ("}", ">", ":"):
                raise ValueError(f"Unexpected textproto token {name}")
            cursor += 1
            if cursor < len(tokens) and tokens[cursor][0] == ":":
                cursor += 1
            token, value_start, _ = tokens[cursor]
            if token in ("{", "<"):
                cursor += 1
                children, end = message("}" if token == "{" else ">")
                fields.append({"name": name, "children": children, "start": start, "end": end})
            else:
                values = []
                while cursor < len(tokens):
                    token, _, end = tokens[cursor]
                    values.append(ast.literal_eval(token) if token.startswith('"') else token)
                    cursor += 1
                    if not token.startswith('"') or cursor >= len(tokens) or not tokens[cursor][0].startswith('"'):
                        break
                fields.append({"name": name, "value": "".join(values), "start": start,
                               "value_start": value_start, "end": end})
        if closing is not None:
            raise ValueError("Unterminated textproto message")
        return fields, len(text)

    return message()[0]


def single(fields, name):
    values = [field for field in fields if field["name"] == name]
    if len(values) != 1:
        raise ValueError(f"Expected exactly one {name} field, found {len(values)}")
    return values[0]


def hlo_expression(text):
    """Normalize names/metadata into a structural expression rooted at the output."""
    nodes = {}
    roots = []
    pattern = re.compile(r'^\s*(ROOT\s+)?%?([\w.\-]+)\s*=\s*([\w]+\[[^\]]*\](?:\{[^}]*\})?)\s+([\w\-]+)\(([^()]*)\)(.*)$')
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        root, name, shape, operation, arguments, attrs = match.groups()
        attrs = attrs.split(", metadata=", 1)[0].split(", backend_config=", 1)[0]
        # Autotune dumps print operand types; optimized HLO prints names only.
        # Array/layout commas belong to the type, not the operand separator.
        parts, start, depth = [], 0, 0
        for index, char in enumerate(arguments):
            if char in "[{": depth += 1
            elif char in "]}": depth -= 1
            elif char == "," and depth == 0:
                parts.append(arguments[start:index]); start = index + 1
        parts.append(arguments[start:])
        operands = []
        for part in parts:
            part = part.strip()
            if not part: continue
            part = re.sub(r"^[\w]+\[[^\]]*\](?:\{[^}]*\})?\s+", "", part)
            operands.append(part.lstrip("%"))
        nodes[name] = {"shape": re.sub(r"\s+", "", shape), "op": operation,
                       "arguments": operands,
                       "attrs": re.sub(r"\s+", "", attrs).lstrip(",")}
        if root:
            roots.append(name)
    if len(roots) != 1:
        raise ValueError(f"Expected a single fusion ROOT, found {len(roots)}")

    def expression(name):
        node = nodes[name]
        arguments = [expression(arg) if arg in nodes else {"literal": arg} for arg in node["arguments"]]
        return {"shape": node["shape"], "op": node["op"], "args": arguments, "attrs": node["attrs"]}

    return expression(roots[0])


def strip_root_views(expression):
    while expression.get("op") in ("bitcast", "reshape"):
        expression = expression["args"][0]
    if "op" not in expression:
        raise ValueError("Fusion output did not resolve to a parsed operation")
    return expression


def autotune_key_expression(expression):
    """XLA's autotune HLO omits constant values; retain their types and positions."""
    result = dict(expression)
    if expression.get("op") == "constant":
        result["args"] = [{"literal": "{...}"}]
    elif "args" in expression:
        result["args"] = [autotune_key_expression(arg) for arg in expression["args"]]
    return result


def constant_payloads(expression, path="root"):
    if expression.get("op") == "constant":
        return [{"path": path, "shape": expression["shape"], "payload": expression["args"]}]
    return [row for index, arg in enumerate(expression.get("args", []))
            for row in constant_payloads(arg, f"{path}/args/{index}")]


def canonical_unsplit_projection(expression, split):
    """Invert only the verified output-projection split-K rewrite (1024→S×K)."""
    if expression["shape"] != f"f32[{split},10,32]{{2,1,0}}" or expression["op"] != "dot":
        raise ValueError("Unexpected canonical split-K output projection")
    left, right = expression["args"]
    expected = (f"f32[10,{split},{1024 // split}]{{2,1,0}}", f"f32[{split},{1024 // split},32]{{2,1,0}}")
    if (left["shape"], right["shape"]) != expected or left["op"] != "bitcast" or right["op"] != "bitcast":
        raise ValueError("Unexpected canonical output-projection split views")
    if expression["attrs"] != "lhs_batch_dims={1},lhs_contracting_dims={2},rhs_batch_dims={0},rhs_contracting_dims={1}":
        raise ValueError("Unexpected split-K projection dimensions")
    return {"shape": "f32[10,32]{1,0}", "op": "dot", "args": [left["args"][0], right["args"][0]],
            "attrs": "lhs_contracting_dims={1},rhs_contracting_dims={0}"}


def function_body(hlo, name):
    match = re.search(r"(?m)^%" + re.escape(name) + r"\s*\([^\n]*\{\n", hlo)
    if not match:
        raise ValueError(f"Canonical fusion not found: {name}")
    end = hlo.index("\n}", match.end())
    return hlo[match.end():end]


def map_configs(dump, canonical_hlo, comparison, output):
    if output.exists():
        raise FileExistsError(output)
    text = dump.read_text()
    top = proto_fields(text)
    entries = [field for field in top if "children" in field
               and any(child["name"] == "hlo" for child in field["children"])]
    if not entries:
        raise ValueError("No AutotuneResults entries with HLO found")
    parsed = []
    for index, entry in enumerate(entries):
        source = single(entry["children"], "hlo")["value"]
        try:
            expression = strip_root_views(hlo_expression(source))
            error = None
        except ValueError as exception:
            expression = None
            error = str(exception)
        parsed.append({"index": index, "entry": entry, "hlo": source, "expression": expression, "parse_error": error})
    review = json.loads(comparison.read_text())
    differences = review["differences_against_canonical"]["full"]
    canonical_rows = {(row["op"], row["occurrence"]): row for row in review["gemm_configs"]["canonical"]}
    if len(differences) != 5:
        raise ValueError("Expected the five independently identified canonical GEMM differences")
    hlo = canonical_hlo.read_text()
    replacements, mappings, used = [], [], set()
    for difference in differences:
        row = canonical_rows[difference["op"], difference["occurrence"]]
        function = re.search(r"calls=%([\w.]+)", row["operation"]).group(1)
        expression = strip_root_views(hlo_expression(function_body(hlo, function)))
        reconstruction = None
        if difference["op"] == "while/body/dot_general" and difference["occurrence"] == 2:
            expression = canonical_unsplit_projection(expression, int(row["config"]["split_k"]))
            reconstruction = "Invert verified split-K8 views into the original1024-axis projection; preserve f32 operand casts and exact dot dimensions"
        key_expression = autotune_key_expression(expression)
        matches = [candidate for candidate in parsed if candidate["expression"] is not None
                   and autotune_key_expression(candidate["expression"]) == key_expression]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous/missing exact normalized fusion match for {difference['op']}: {len(matches)}")
        match = matches[0]
        if match["index"] in used:
            raise ValueError("Two canonical operations mapped to the same autotune entry")
        used.add(match["index"])
        result = single(match["entry"]["children"], "result")
        triton = single(result["children"], "triton")
        previous = {field["name"]: field["value"] for field in triton["children"]}
        replacement = "triton {\n" + "".join(f"        {name}: {int(value)}\n" for name, value in row["config"].items()) + "      }"
        replacements.append((triton["start"], triton["end"], replacement))
        removed_timing = []
        for field in result["children"]:
            if field["name"] == "run_time":
                replacements.append((field["start"], field["end"], ""))
                removed_timing.append(text[field["start"]:field["end"]])
        mappings.append({"operation": difference["op"], "occurrence": difference["occurrence"],
                         "canonical_line": row["line"], "canonical_fusion": function,
                         "autotune_entry": match["index"], "hlo_sha256": hashlib.sha256(match["hlo"].encode()).hexdigest(),
                         "normalized_expression": expression, "projection_reconstruction": reconstruction,
                         "canonical_constant_payloads": constant_payloads(expression),
                         "dump_constant_payloads": constant_payloads(match["expression"]),
                         "old_config": previous, "canonical_config": row["config"], "removed_old_config_timing": removed_timing})
    rewritten = text
    for start, end, replacement in sorted(replacements, reverse=True):
        rewritten = rewritten[:start] + replacement + rewritten[end:]
    # Parse the full product before publishing; numerical replay is a separate gate.
    proto_fields(rewritten)
    output.mkdir(parents=True)
    target = output / "canonical_autotune.textproto"
    target.write_text(rewritten)
    audit = {"complete": True, "mapping_passed": True, "numerical_fidelity_proven": False,
             "scope": "Offline compiler-configuration mapping only; original dump/shared caches untouched. Replay against saved native actions remains mandatory.",
             "sources": {str(path.resolve()): sha(path) for path in (dump, canonical_hlo, comparison, Path(__file__))},
             "entries": len(entries), "matched_and_rewritten_entries": len(mappings), "mappings": mappings,
             "output": {"path": str(target.resolve()), "sha256": sha(target)},
             "constant_scope": "XLA autotune HLO elides constant payloads as {...}; matching preserves constant dtype, shape, position and surrounding operations, but cannot establish omitted literal equality. Canonical literal values and dump omissions are recorded per match. Numerical oracle replay is mandatory.",
             "timing_scope": "Old autotuned run_time removed from modified entries; no timing for overridden configurations is invented."}
    (output / "mapping_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"mapping_passed": True, "matched": len(mappings), "output": str(target)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--canonical-hlo", required=True, type=Path)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    map_configs(args.dump, args.canonical_hlo, args.comparison, args.output)

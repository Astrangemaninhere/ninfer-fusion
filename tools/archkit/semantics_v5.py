import ast
import io
import urllib.request


def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'ninfer-tools'})
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.read().decode('utf-8', 'replace')


def derive(model_type: str, knobs: dict) -> dict:
    base = ('https://raw.githubusercontent.com/huggingface/transformers/main/'
            'src/transformers/models/')
    name = model_type.replace('_text', '')
    src = None
    for u in [base + name + '/modeling_' + name + '.py',
              base + model_type + '/modeling_' + model_type + '.py']:
        try:
            src = fetch(u)
            break
        except Exception:
            continue
    if src is None:
        return {'error': 'reference not found'}
    lines = src.splitlines()
    tree = ast.parse(src)
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in knobs:
            ln = max(0, node.lineno - 1)
            win = lines[max(0, ln - 2):ln + 3]
            out.setdefault(node.attr, []).append(' | '.join(
                w.strip()[:120] for w in win))
    return {k: v[:4] for k, v in out.items()}

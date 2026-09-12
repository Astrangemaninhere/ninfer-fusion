import json
d = json.load(open(r'C:\Users\User\Documents\ziqinzhang\_vllm_tree.json', encoding='utf-8'))
paths = [t['path'] for t in d.get('tree', []) if 'dflash' in t['path'].lower() or 'lookup' in t['path'].lower() or 'ngram' in t['path'].lower()]
for p in paths:
    print(p)

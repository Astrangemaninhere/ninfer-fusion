# -*- coding: utf-8 -*-
"""kv_calibrate.py — KV 量化程度校准器: 扫描 E8 覆盖率, 输出质量-压缩曲线.

对每个覆盖点:
  1. 以 N 层 E8 + (16-N) 层 NVFP4 启动 serve
  2. 跑针尖测试 (57K 上下文, 精确召回)
  3. 记录 (覆盖率, PASS/FAIL, 响应时间)
输出: 质量-压缩曲线 + 推荐安全上限。
用法: python3 kv_calibrate.py [start_N] [end_N] [step]
"""
import json
import subprocess
import sys
import time

CORPUS = ('The history of thermodynamics spans centuries. Early studies of heat and work '
          'led to the caloric theory, later replaced by the mechanical theory of heat. '
          'James Prescott Joule measured the mechanical equivalent of heat through careful '
          'experiments with paddle wheels and falling weights. Lord Kelvin formalized the '
          'absolute temperature scale. Rudolf Clausius introduced the concept of entropy, '
          'and Ludwig Boltzmann connected entropy to microscopic disorder through statistical '
          'mechanics. The laws of thermodynamics govern engines, refrigerators, and the '
          'arrow of time itself. Every working engineer relies on these principles daily. ')
NEEDLE = 'The secret code is XKCD-42-7777. Remember it carefully.'
PROMPT = (NEEDLE + ' ' + CORPUS * 520 +
          '\n\nNow answer: what is the secret code mentioned at the very beginning? '
          'Also, name two scientists from the passage and their contributions.')
PORT = 8003
MODEL = '/home/user/models/qwen3_8_27b_nvfp4.ninfer'


def start_serve(n_e8: int) -> bool:
    subprocess.run(['bash', '-c',
                    'for pid in $(pgrep -f ninfer-serve); do kill -9 $pid 2>/dev/null; done; sleep 2'],
                   check=False)
    spec = f'--kv-layer-storage 0-{n_e8 - 1}:e8' if n_e8 > 0 else ''
    cmd = (f'nohup env LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64 '
           f'{MODEL.replace("/models/", "/ninfer-fusion/build/apps/").replace("qwen3_8_27b_nvfp4.ninfer", "ninfer-serve")} '
           f'{MODEL} --port {PORT} --kv-dtype nvfp4 {spec} '
           f'--max-context 65536 --kv-capacity 65536 --greedy --max-concurrency 1 '
           f'> /home/user/serve_cal.log 2>&1 &')
    subprocess.run(['bash', '-c', cmd], check=False)
    for _ in range(30):
        time.sleep(5)
        try:
            log = open('/home/user/serve_cal.log', encoding='utf-8', errors='replace').read()
            if 'listening' in log:
                return True
        except FileNotFoundError:
            pass
    return False


def needle_test() -> tuple[bool, float, str]:
    payload = {'model': 'qwen3.8-27b',
               'messages': [{'role': 'user', 'content': PROMPT}],
               'max_tokens': 400, 'temperature': 0.0}
    open('/tmp/cal_req.json', 'w').write(json.dumps(payload))
    t0 = time.time()
    out = subprocess.run(['curl', '-s', '-m', '600', '-X', 'POST',
                          f'http://127.0.0.1:{PORT}/v1/chat/completions',
                          '-H', 'Content-Type: application/json', '-d', '@/tmp/cal_req.json'],
                         capture_output=True, text=True, timeout=620).stdout
    dt = time.time() - t0
    try:
        d = json.loads(out)
        text = d['choices'][0]['message'].get('content', '') + ' ' + \
            d['choices'][0]['message'].get('reasoning_content', '')
        return 'XKCD-42-7777' in text, dt, text[:150].replace(chr(10), ' ')
    except Exception:
        return False, dt, 'PARSE_FAIL'


def main():
    start_n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end_n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    results = []
    for n in range(start_n, end_n + 1, step):
        print(f'--- N_E8={n} coverage={n/16*100:.0f}%', flush=True)
        if not start_serve(n):
            print('  serve FAILED')
            results.append((n, None, None))
            continue
        hit, dt, head = needle_test()
        print(f'  hit={hit} t={dt:.0f}s head: {head}')
        results.append((n, hit, dt))
    print('\n== 质量-压缩曲线 ==')
    print('E8层数 | 覆盖率 | 针尖 | 响应s')
    print('-------+--------+------+------')
    for n, hit, dt in results:
        cov = f'{n/16*100:.0f}%'
        print(f'{n:5d}  | {cov:6s} | {"PASS" if hit else "FAIL":4s} | {dt}')
    # 找最优
    best = max((n for n, h, _ in results if h), default=0)
    print(f'\n推荐: E8 ≤ {best} 层 (覆盖率 {best/16*100:.0f}%), 其余 NVFP4')
    print(f'对应参数: --kv-dtype nvfp4 --kv-layer-storage 0-{best-1}:e8')


if __name__ == '__main__':
    main()

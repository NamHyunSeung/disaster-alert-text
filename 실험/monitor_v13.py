"""
v13 학습 로그 모니터 - MILESTONE 및 에폭 요약 라인만 출력
"""
import time, re, sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

LOG = r"d:\Users\Namhyeonseung\Desktop\파일\대학 수업\3학년\1학기\딥러닝 기초\프로젝트\results\train_log_model_v13.txt"

seen = set()

def read_log():
    for enc in ['utf-8', 'utf-16', 'utf-16-le']:
        try:
            with open(LOG, encoding=enc, errors='replace') as f:
                return f.read()
        except Exception:
            continue
    return ''

while True:
    try:
        content = read_log()
        for line in content.splitlines():
            line = line.strip()
            if not line or line in seen:
                continue

            is_milestone     = '[MILESTONE]' in line
            is_epoch_summary = bool(re.search(r'Epoch \d+/\d+ \|.*Val', line))
            is_upsample      = 'Synthetic upsampling' in line
            is_criterion     = 'Criterion:' in line
            is_fatal         = any(x in line for x in ['Traceback', 'Error:', 'CUDA out', 'Killed', 'OOM'])

            if is_milestone or is_epoch_summary or is_upsample or is_criterion or is_fatal:
                seen.add(line)
                print(line, flush=True)

                if is_epoch_summary and re.search(r'Epoch 20/20', line):
                    sys.exit(0)

    except Exception as e:
        print(f"[MONITOR ERROR] {e}", flush=True)

    time.sleep(3)

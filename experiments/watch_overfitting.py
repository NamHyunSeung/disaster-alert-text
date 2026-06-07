import re
import time
import sys

LOG_FILE = r"d:\Users\Namhyeonseung\Desktop\파일\대학 수업\3학년\1학기\딥러닝 기초\프로젝트\results\train_v22_log.txt"
OUTPUT_FILE = r"d:\Users\Namhyeonseung\Desktop\파일\대학 수업\3학년\1학기\딥러닝 기초\프로젝트\results\overfitting_v22.txt"

EPOCH_PATTERN = re.compile(
    r'Epoch (\d+)/\d+ \| Train Loss: ([\d.]+) \| Val Loss: ([\d.]+) \| Val Acc: ([\d.]+)% \| Val Macro F1: ([\d.]+)% \| Masked MacroF1: ([\d.]+)%.*?'
    r'L4 F1=([\d.]+)%/R=([\d.]+)%'
)

def parse_epochs(log_file):
    with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    return EPOCH_PATTERN.findall(content)

def l4_precision(f1, recall):
    f1, recall = float(f1), float(recall)
    if 2 * recall - f1 <= 0:
        return 0.0
    return f1 * recall / (2 * recall - f1)

def write_table(epochs):
    lines = []
    lines.append("="*95)
    lines.append(f"{'Ep':>3} | {'TrainL':>7} | {'ValL':>6} | {'Gap':>7} | {'ValF1':>7} | {'MaskedF1':>9} | {'L4 F1':>6} | {'L4 R':>6} | {'L4 P(추정)':>10} | 판정")
    lines.append("-"*95)

    prev_val_loss = None
    prev_val_f1 = None
    for ep in epochs:
        epoch, train_loss, val_loss, val_acc, val_f1, masked_f1, l4_f1, l4_r = ep
        tl, vl = float(train_loss), float(val_loss)
        gap = tl - vl
        l4_p = l4_precision(l4_f1, l4_r)

        flags = []
        if prev_val_loss is not None and vl > prev_val_loss + 0.0005:
            flags.append("Val Loss↑")
        if prev_val_f1 is not None and float(val_f1) < float(prev_val_f1) - 0.1:
            flags.append("F1 저하")
        status = "  ⚠ " + ", ".join(flags) if flags else "  정상"

        lines.append(
            f"{epoch:>3} | {tl:>7.4f} | {vl:>6.4f} | {gap:>+7.4f} | {val_f1:>6}% | {masked_f1:>8}% | {l4_f1:>5}% | {l4_r:>5}% | {l4_p:>9.1f}% |{status}"
        )
        prev_val_loss = vl
        prev_val_f1 = val_f1

    lines.append("="*95)
    lines.append(f"마지막 업데이트: {time.strftime('%Y-%m-%d %H:%M:%S')}  (총 {len(epochs)} epoch)")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

    for line in lines:
        print(line)

print(f"Watching: {LOG_FILE}")
print(f"Output:   {OUTPUT_FILE}")
print("Ctrl+C로 종료\n")

last_count = 0
while True:
    try:
        epochs = parse_epochs(LOG_FILE)
        if len(epochs) > last_count:
            last_count = len(epochs)
            write_table(epochs)
            print()
        time.sleep(30)
    except KeyboardInterrupt:
        print("\n종료.")
        sys.exit(0)
    except Exception as e:
        print(f"[오류] {e}")
        time.sleep(30)

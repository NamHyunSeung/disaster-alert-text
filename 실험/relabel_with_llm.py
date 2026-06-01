"""
LLM 보조 재레이블링 — 맥락 기반 (키워드 독립)
대상: L2/L3/L4 전체
방법: claude CLI subprocess로 마스킹 텍스트 재레이블링
"""

import json
import subprocess
import time
import sys
import re
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, '.')
from dataset_v2 import _mask_text
from dataset import preprocess_text

# ─── 설정 ─────────────────────────────────────────────
DATA_IN  = '중요파일/data/raw/재난문자_레이블링결과_dedup_v3.xlsx'
DATA_OUT = '중요파일/data/raw/재난문자_레이블링결과_dedup_v4_llm.xlsx'
PROGRESS = '중요파일/data/raw/relabel_progress_v4.json'
TARGET   = [2, 3, 4]
BATCH    = 20
DELAY    = 0.0
WORKERS  = 8   # 병렬 subprocess 수
CLAUDE   = r'C:\Users\namzx\AppData\Roaming\npm\node_modules\@anthropic-ai\claude-code\bin\claude.exe'

SYSTEM = """재난문자 레이블 기준 (키워드 무관 — 상황 심각도 기반):

L0(긴급아님): 훈련/행사/완료/해제 알림, 일반 행정 공지
L1(낮음): 정보성 안내, 잠재 위험 예보 수준, 별도 행동 불필요
L2(중간): 행정 조치 시행 — 집합금지, 시설폐쇄, 방역협조, 출입제한(행정). 즉각 생명위험 없음.
L3(높음): 기상·재난 상황 진행 중. 야외자제·동파대비·농작물피해대비·체온유지 등 예방적 행동 요구.
L4(매우높음): 즉각 안전행동 필요. 특정 장소로 이동 지시, 위험구역 접근금지(생명안전), 즉각 대피.

핵심 규칙:
- "대피", "경보", "주의보", "특보", "화재", "홍수" 등 긴급도 키워드 무시
- 특정 장소로 이동 지시 → L4 / 예방적 조치 권고 → L3 / 행정 제한 조치 → L2
반드시 JSON 배열로만 응답. 설명 없음. 예: [{"idx":0,"label":3},{"idx":1,"label":2}]"""


def build_prompt(batch: list[dict]) -> str:
    lines = [SYSTEM, "\n다음 재난문자들의 레이블을 판정하세요:\n"]
    for item in batch:
        lines.append(f'[{item["idx"]}]')
        lines.append(f'원본: {item["orig"][:150]}')
        lines.append(f'마스킹: {item["masked"][:150]}')
        lines.append('')
    return '\n'.join(lines)


def call_claude(prompt: str, retries: int = 3) -> list[dict]:
    for attempt in range(retries):
        try:
            result = subprocess.run(
                [CLAUDE, '-p', prompt],
                capture_output=True, text=True, encoding='utf-8',
                stdin=subprocess.DEVNULL, timeout=90
            )
            text = result.stdout.strip()
            # ```json 블록 제거
            if '```' in text:
                inner = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
                if inner:
                    text = inner.group(1).strip()
            # JSON 배열 추출
            match = re.search(r'\[[\s\S]*\]', text)
            if match:
                parsed = json.loads(match.group())
                # "index" 키를 "idx"로 통일
                return [{'idx': r.get('idx', r.get('index', -1)), 'label': r['label']} for r in parsed]
            # 단일 객체 시도
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                r = json.loads(match.group())
                return [{'idx': r.get('idx', r.get('index', -1)), 'label': r['label']}]
        except (json.JSONDecodeError, subprocess.TimeoutExpired, Exception) as e:
            print(f"\n  [재시도 {attempt+1}/{retries}] {type(e).__name__}: {e}")
            time.sleep(2 ** attempt)
    return []


def main():
    print("데이터 로드 중...")
    df = pd.read_excel(DATA_IN)
    df = df[df['label'] != -1].copy()
    df['label'] = df['label'].astype(int)
    df['text'] = df['메시지내용'].fillna('').apply(preprocess_text)
    df['masked'] = df['text'].apply(_mask_text)
    df['new_label'] = df['label'].copy()

    target_df = df[df['label'].isin(TARGET)].copy()
    n2 = (target_df['label']==2).sum()
    n3 = (target_df['label']==3).sum()
    n4 = (target_df['label']==4).sum()
    print(f"재레이블링 대상: {len(target_df):,}건 (L2:{n2:,} / L3:{n3:,} / L4:{n4:,})")

    # 진행 상황 로드
    done: dict[int, int] = {}
    if __import__('os').path.exists(PROGRESS):
        with open(PROGRESS, encoding='utf-8') as f:
            done = {int(k): v for k, v in json.load(f).items()}
        print(f"이전 진행 로드: {len(done):,}건 완료")

    indices = [i for i in target_df.index if i not in done]
    print(f"남은 작업: {len(indices):,}건 ({len(indices)//BATCH + 1}배치)")

    batches = [indices[i:i+BATCH] for i in range(0, len(indices), BATCH)]
    errors = 0
    lock = __import__('threading').Lock()
    start_time = time.time()

    def process_batch(args):
        batch_num, batch_idxs = args
        batch_data = [
            {'idx': i, 'orig': str(df.loc[i, 'text'])[:120], 'masked': str(df.loc[i, 'masked'])[:120]}
            for i in batch_idxs
        ]
        results = call_claude(build_prompt(batch_data))
        return batch_num, batch_idxs, results

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process_batch, (bn, bi)): bn for bn, bi in enumerate(batches)}
        pbar = tqdm(total=len(indices), desc="재레이블링", unit="건")
        completed_batches = 0

        for future in as_completed(futures):
            batch_num, batch_idxs, results = future.result()
            with lock:
                if not results:
                    errors += len(batch_idxs)
                    for i in batch_idxs:
                        done[i] = int(df.loc[i, 'label'])
                else:
                    result_map = {r['idx']: r['label'] for r in results}
                    for i in batch_idxs:
                        done[i] = result_map.get(i, int(df.loc[i, 'label']))

                pbar.update(len(batch_idxs))
                completed_batches += 1

                if completed_batches % 25 == 0:
                    with open(PROGRESS, 'w', encoding='utf-8') as f:
                        json.dump(done, f)
                    elapsed = time.time() - start_time
                    done_cnt = len(done)
                    remaining = len(indices) - done_cnt
                    eta = (elapsed / done_cnt * remaining) if done_cnt else 0
                    changed_so_far = sum(1 for i, v in done.items() if i in target_df.index and v != df.loc[i, 'label'])
                    tqdm.write(f"  [{done_cnt:,}/{len(indices):,}] 변경:{changed_so_far}건 ETA:{eta/3600:.1f}h")
        pbar.close()
        print(f"\n완료. 총 소요: {(time.time()-start_time)/3600:.1f}h / 오류: {errors}건")

    # 최종 저장
    with open(PROGRESS, 'w', encoding='utf-8') as f:
        json.dump(done, f)

    # new_label 반영
    for idx, new_lbl in done.items():
        df.loc[idx, 'new_label'] = new_lbl

    # 변경 통계
    target_mask = df['label'].isin(TARGET)
    changed = df[target_mask & (df['label'] != df['new_label'])]
    print(f"\n{'='*60}")
    print(f"총 변경: {len(changed):,}건 / 대상 {len(target_df):,}건 ({len(changed)/len(target_df)*100:.1f}%)")
    print(f"API 오류: {errors}건")

    from collections import Counter
    NAMES = {0:'긴급아님', 1:'낮음', 2:'중간', 3:'높음', 4:'매우높음'}
    print(f"\n레이블 변화 분포:")
    for (orig, new), cnt in sorted(Counter(zip(changed['label'], changed['new_label'])).items()):
        print(f"  L{orig}({NAMES[orig]}) → L{new}({NAMES[new]}) : {cnt:,}건")

    print(f"\n전체 레이블 분포 (변경 전 → 변경 후):")
    for lbl in range(5):
        before = (df['label'] == lbl).sum()
        after  = (df['new_label'] == lbl).sum()
        print(f"  L{lbl} {NAMES[lbl]:<5}: {before:>7,} → {after:>7,} ({after-before:>+6,})")

    # label → new_label로 교체 후 저장
    df_out = df.drop(columns=['text', 'masked', 'label']).rename(columns={'new_label': 'label'})
    df_out.to_excel(DATA_OUT, index=False)
    print(f"\n저장 완료: {DATA_OUT}")


if __name__ == '__main__':
    main()

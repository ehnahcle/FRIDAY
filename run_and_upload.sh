#!/bin/bash
# run_and_upload.sh
# FRIDAY (SPACE) 스크리너 실행 후 GitHub 업로드

set -e

cd ~/Documents/friday

# ============================================================
# Pre-flight: 운영 변경사항 unstaged면 즉시 중단
# 2026-05-26 quant_tool incident 패턴 답습 — 6일간 unstaged로 GHA broken
# 생성될 CSV/macro 파일은 예외, 그 외는 commit 후 실행
# ============================================================
echo "🔍 Pre-flight: 운영 변경사항 검사 ..."
DIRTY=$(git status --porcelain | grep -v -E ' (friday_results\.csv|macro_indicators\.json|volmanage_state\.json)$' || true)
if [ -n "$DIRTY" ]; then
    echo ""
    echo "❌ 커밋 안 된 운영 변경사항 발견 — friday 실행 중단."
    echo ""
    echo "$DIRTY"
    echo ""
    echo "처리: 위 파일들을 commit 또는 stash 후 재실행."
    exit 1
fi
echo "  ✓ Working tree clean (생성 파일만 수정될 예정)"
echo ""

echo "🚀 Step 1/3: 가상환경 활성화"
# venv는 quant_tool 의 것 재사용 (의존성 동일)
if [ -d ~/Documents/friday/venv ]; then
    source ~/Documents/friday/venv/bin/activate
elif [ -d ~/Documents/quant_tool/venv ]; then
    source ~/Documents/quant_tool/venv/bin/activate
else
    echo "❌ venv 없음. python -m venv venv && pip install -r requirements.txt"
    exit 1
fi

echo ""
echo "🚀 Step 2/3: SPACE 스크리너 실행 (R1000 + ADV20 \$5M gate)"
python screener.py

echo ""
echo "☁️  Step 3/3: GitHub 업로드"
# 명시적 git add — 5-26 incident 재발 방지
git add friday_results.csv macro_indicators.json
# volmanage_state.json은 vol_manage.py가 생성 시에만 추가
if [ -f volmanage_state.json ]; then
    git add volmanage_state.json
fi
git commit -m "Update results: $(date '+%Y-%m-%d %H:%M')" || echo "No changes"
git push

echo ""
echo "✅ 완료!"

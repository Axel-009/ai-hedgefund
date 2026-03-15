#!/usr/bin/env bash
# =============================================================================
# AI Hedge Fund — Daily Cron Scheduler
# Run once to install the 3 daily jobs into crontab.
#
# Jobs (all times ET = UTC-4 during summer, UTC-5 during winter):
#   09:30 ET = 13:30 UTC  → run_open.py
#   Every hour 10:30–15:30 ET = 14:30–19:30 UTC → run_hourly.py
#   16:00 ET = 20:00 UTC  → run_close.py
#
# Usage:
#   bash setup_cron.sh            # install jobs
#   bash setup_cron.sh --remove   # remove jobs
#   bash setup_cron.sh --status   # show current crontab
# =============================================================================

set -euo pipefail

BASE=/home/user/ai-hedgefund
PYTHON=$(command -v python3)
LOG=/home/user/ai-hedgefund/logs/cron.log

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

case "${1:-}" in
  --remove)
    echo -e "${CYAN}Removing hedge fund cron jobs...${NC}"
    crontab -l 2>/dev/null | grep -v "ai-hedgefund" | crontab -
    echo -e "${GREEN}✓ Removed.${NC}"
    exit 0
    ;;
  --status)
    echo -e "${BOLD}Current crontab:${NC}"
    crontab -l 2>/dev/null || echo "(empty)"
    exit 0
    ;;
esac

echo -e "\n${BOLD}Installing AI Hedge Fund cron jobs...${NC}"

# Preserve existing crontab entries (excluding any prior ai-hedgefund entries)
EXISTING=$(crontab -l 2>/dev/null | grep -v "ai-hedgefund" || true)

NEW_JOBS=$(cat <<CRON
# ── AI Hedge Fund Daily Jobs ──────────────────────────────────────────────────
# Open: 9:30 ET = 13:30 UTC (Mon–Fri)
30 13 * * 1-5 cd ${BASE} && ${PYTHON} run_open.py >> ${LOG} 2>&1   # ai-hedgefund open

# Hourly: 10:30–15:30 ET = 14:30–19:30 UTC (Mon–Fri)
30 14 * * 1-5 cd ${BASE} && ${PYTHON} run_hourly.py >> ${LOG} 2>&1  # ai-hedgefund hourly
30 15 * * 1-5 cd ${BASE} && ${PYTHON} run_hourly.py >> ${LOG} 2>&1  # ai-hedgefund hourly
30 16 * * 1-5 cd ${BASE} && ${PYTHON} run_hourly.py >> ${LOG} 2>&1  # ai-hedgefund hourly
30 17 * * 1-5 cd ${BASE} && ${PYTHON} run_hourly.py >> ${LOG} 2>&1  # ai-hedgefund hourly
30 18 * * 1-5 cd ${BASE} && ${PYTHON} run_hourly.py >> ${LOG} 2>&1  # ai-hedgefund hourly
30 19 * * 1-5 cd ${BASE} && ${PYTHON} run_hourly.py >> ${LOG} 2>&1  # ai-hedgefund hourly

# Close: 4:00 PM ET = 20:00 UTC (Mon–Fri)
0 20 * * 1-5  cd ${BASE} && ${PYTHON} run_close.py >> ${LOG} 2>&1   # ai-hedgefund close

# Earnings graph: refresh every 5 min during market hours (13:30–20:05 UTC Mon–Fri)
*/5 13-20 * * 1-5 cd ${BASE} && ${PYTHON} src/reporting/live_earnings_graph.py >> ${LOG} 2>&1  # ai-hedgefund graph
CRON
)

# Install combined crontab
printf "%s\n%s\n" "$EXISTING" "$NEW_JOBS" | crontab -

echo -e "${GREEN}✓ Cron jobs installed:${NC}"
echo ""
echo "  09:30 ET  → run_open.py    (morning scan + execute signals)"
echo "  10:30–15:30 ET → run_hourly.py  (6 × intraday updates)"
echo "  16:00 ET  → run_close.py   (EOD reconciliation + model training)"
echo "  Every 5 min (market hours) → live_earnings_graph.py"
echo ""
echo -e "  Logs: ${CYAN}${LOG}${NC}"
echo -e "  Chart: ${CYAN}${BASE}/reports/earnings_graph_\$(date +%F).html${NC}"
echo ""
echo -e "${BOLD}To view installed jobs:${NC}  bash setup_cron.sh --status"
echo -e "${BOLD}To remove all jobs:${NC}     bash setup_cron.sh --remove"
echo ""
echo -e "${RED}NOTE:${NC} Times above assume UTC server timezone."
echo "  If your server is in ET, change 13:30/14:30 etc. → 9:30/10:30 etc."
echo "  Check with: date +%Z"
echo ""

#!/usr/bin/env bash
# =============================================================================
# AI Hedge Fund — Full Environment Setup
# Run this at the start of every new session from /home/user/ai-hedgefund/
#
# What it does:
#   1. Clones / updates all required repos into /home/user/
#   2. Checks out correct branch (claude/init-test-repos-oPogr) where applicable
#   3. Creates .env from .env.example if missing
#   4. Installs Python dependencies
#   5. Runs health check (init_test_all.py)
# =============================================================================

set -euo pipefail
BASE=/home/user

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}!${NC} $*"; }
fail() { echo -e "${RED}✗${NC} $*"; }
info() { echo -e "${CYAN}→${NC} $*"; }

BRANCH="claude/init-test-repos-oPogr"
PROXY="http://local_proxy@127.0.0.1:34551/git/Axel-009"

echo -e "\n${BOLD}═══════════════════════════════════════════${NC}"
echo -e "${BOLD}  AI Hedge Fund — Environment Setup        ${NC}"
echo -e "${BOLD}═══════════════════════════════════════════${NC}\n"

# =============================================================================
# REPO DEFINITIONS
# Format: "local_name|remote_url|branch_to_checkout"
# =============================================================================

declare -a REPOS=(
  # ── Internal Axel-009 repos (local proxy) ──────────────────────────────────
  "ai-hedgefund|${PROXY}/ai-hedgefund|${BRANCH}"
  "Financial-Data|${PROXY}/Financial-Data|${BRANCH}"
  "ML-Macro-Market|${PROXY}/ML-Macro-Market|${BRANCH}"
  "Mav-Analysis|${PROXY}/Mav-Analysis|${BRANCH}"
  "QLIB|${PROXY}/QLIB|${BRANCH}"
  "Ruflo-agents|${PROXY}/Ruflo-agents|${BRANCH}"
  "AI-Newton|${PROXY}/AI-Newton|${BRANCH}"
  "Air-LLM|${PROXY}/Air-LLM|${BRANCH}"
  "hedgefund-tracker|${PROXY}/hedgefund-tracker|${BRANCH}"
  "open-bb|${PROXY}/open-bb|${BRANCH}"
  "quant-trading|${PROXY}/quant-trading|${BRANCH}"

  # ── External public repos (GitHub) ─────────────────────────────────────────
  # HFT execution engine sources
  "Stock-Prediction-Models|https://github.com/huseinzol05/Stock-Prediction-Models.git|master"
  "FinRL|https://github.com/AI4Finance-Foundation/FinRL.git|master"
  "Deep-Trading|https://github.com/Rachnog/Deep-Trading.git|master"
  "DeepLearningExamples|https://github.com/NVIDIA/DeepLearningExamples.git|master"
  "DeepLearning|https://github.com/Mikoto10032/DeepLearning.git|master"
  "kserve|https://github.com/kserve/kserve.git|master"
  "wondertrader|https://github.com/wondertrader/wondertrader.git|master"
  "exchange-core|https://github.com/exchange-core/exchange-core.git|master"

  # ARIMA + Laplacian MC gist (Monte Carlo bridge)
  "gist-b16f9d8cd0a9e817fd3baa3ce3cd0194|https://gist.github.com/b16f9d8cd0a9e817fd3baa3ce3cd0194.git|master"

  # Top-down / bottom-up methodology reference
  "building-stock-analysis|https://github.com/MODERATE-Project/building-stock-analysis.git|main"

  # Macro / data layer
  "FRB|https://github.com/avelkoski/FRB.git|master"
  "Quant-Developers-Resources|https://github.com/cybergeekgyan/Quant-Developers-Resources.git|main"
)

# =============================================================================
# STEP 1 — Clone or update each repo
# =============================================================================
echo -e "${BOLD}[1/5] Repositories${NC}"

CLONE_OK=0; CLONE_SKIP=0; CLONE_FAIL=0

for entry in "${REPOS[@]}"; do
  IFS='|' read -r name url branch <<< "$entry"
  dest="${BASE}/${name}"

  if [[ -d "${dest}/.git" ]]; then
    # Already cloned — fetch + checkout correct branch
    cur=$(git -C "$dest" branch --show-current 2>/dev/null || echo "DETACHED")
    if [[ "$cur" != "$branch" ]]; then
      git -C "$dest" fetch origin "$branch" --quiet 2>/dev/null \
        && git -C "$dest" checkout "$branch" --quiet 2>/dev/null \
        || warn "$name: could not switch to $branch (staying on $cur)"
    fi
    git -C "$dest" pull --ff-only --quiet 2>/dev/null \
      && ok "$name (updated, branch: $branch)" \
      || warn "$name: pull failed — using existing state"
    CLONE_SKIP=$((CLONE_SKIP + 1))
  else
    info "Cloning $name ..."
    if git clone --branch "$branch" --depth 1 "$url" "$dest" --quiet 2>/dev/null; then
      ok "$name cloned → ${dest}"
      CLONE_OK=$((CLONE_OK + 1))
    else
      # Retry without --branch (some repos may not have the branch yet)
      if git clone --depth 1 "$url" "$dest" --quiet 2>/dev/null; then
        warn "$name cloned (no branch $branch — using default)"
        CLONE_OK=$((CLONE_OK + 1))
      else
        fail "$name: clone FAILED from $url"
        CLONE_FAIL=$((CLONE_FAIL + 1))
      fi
    fi
  fi
done

echo -e "  ${GREEN}${CLONE_OK} cloned${NC}  ${CYAN}${CLONE_SKIP} updated${NC}  ${RED}${CLONE_FAIL} failed${NC}\n"

# =============================================================================
# STEP 2 — .env file
# =============================================================================
echo -e "${BOLD}[2/5] Environment variables${NC}"
ENV_FILE="${BASE}/ai-hedgefund/.env"
ENV_EXAMPLE="${BASE}/ai-hedgefund/.env.example"

if [[ -f "$ENV_FILE" ]]; then
  ok ".env already exists"
else
  if [[ -f "$ENV_EXAMPLE" ]]; then
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    warn ".env created from .env.example — fill in your API keys:"
    echo "     ${ENV_FILE}"
    echo ""
    echo "     Required keys:"
    echo "       ANTHROPIC_API_KEY      — https://console.anthropic.com/"
    echo "       OPENAI_API_KEY         — https://platform.openai.com/"
    echo "       FINANCIAL_DATASETS_API_KEY — https://financialdatasets.ai/"
  else
    fail ".env.example not found — create .env manually"
  fi
fi
echo ""

# =============================================================================
# STEP 3 — Python dependencies
# =============================================================================
echo -e "${BOLD}[3/5] Python dependencies${NC}"
cd "${BASE}/ai-hedgefund"

if command -v poetry &>/dev/null; then
  info "Installing via poetry..."
  poetry install --no-interaction --quiet 2>&1 | tail -3
  ok "poetry install complete"
elif command -v pip &>/dev/null; then
  info "poetry not found — trying pip install from pyproject.toml..."
  pip install -e . --quiet 2>&1 | tail -3 \
    && ok "pip install complete" \
    || warn "pip install had issues — check manually"
else
  fail "Neither poetry nor pip found"
fi

# Core runtime deps that bridges need (graceful — don't abort on failure)
info "Checking optional ML deps..."
python3 -c "import numpy, pandas" 2>/dev/null      && ok "numpy + pandas" || warn "numpy/pandas missing — pip install numpy pandas"
python3 -c "import sklearn"       2>/dev/null      && ok "scikit-learn"   || warn "scikit-learn missing — pip install scikit-learn"
python3 -c "import xgboost"       2>/dev/null      && ok "xgboost"        || warn "xgboost missing (optional) — pip install xgboost"
python3 -c "import yfinance"      2>/dev/null      && ok "yfinance"       || warn "yfinance missing — pip install yfinance"
python3 -c "import torch"         2>/dev/null      && ok "PyTorch"        || warn "PyTorch missing (optional for DRL) — pip install torch"
echo ""

# =============================================================================
# STEP 4 — Smoke-test core imports
# =============================================================================
echo -e "${BOLD}[4/5] Import health check${NC}"
cd "${BASE}/ai-hedgefund"

python3 - <<'PYCHECK'
import sys, os
sys.path.insert(0, '.')
sys.path.insert(0, 'src')

results = []

checks = [
    ("ExecutionEngine",     "from src.agents.execution_engine import ExecutionEngine, SignalType"),
    ("SignalType (15)",     "from src.agents.execution_engine import SignalType; assert len(list(SignalType)) == 15"),
    ("StockPredictionBridge", "from src.models.stock_prediction_bridge import StockPredictionBridge"),
    ("FinRLBridge",         "from src.models.finrl_bridge import FinRLBridge"),
    ("DeepTradingFeatures", "from src.models.deep_trading_features import DeepTradingFeatures"),
    ("NVIDIATFTAdapter",    "from src.models.nvidia_tft_adapter import NVIDIATFTAdapter"),
    ("KServeAdapter",       "from src.models.kserve_adapter import KServeAdapter"),
    ("MonteCarloBridge",    "from src.models.monte_carlo_bridge import MonteCarloBridge"),
    ("UniverseClassifier",  "from src.models.universe_classifier import UniverseClassifier, FundamentalsStore"),
    ("ModelEvaluator",      "from src.models.model_evaluator import ModelEvaluator"),
    ("MacroEngine",         "from src.agents.macro_engine import MacroEngine"),
    ("AlphaOptimizer",      "from src.agents.alpha_optimizer import AlphaOptimizer"),
]

for name, stmt in checks:
    try:
        exec(stmt)
        print(f"  \033[32m✓\033[0m {name}")
    except Exception as e:
        print(f"  \033[31m✗\033[0m {name}: {e}")

PYCHECK

echo ""

# =============================================================================
# STEP 5 — Full repo health check
# =============================================================================
echo -e "${BOLD}[5/5] Repository inventory${NC}"
cd "${BASE}/ai-hedgefund"
python3 init_test_all.py 2>/dev/null | grep -E "(✓|✗|PASS|FAIL|Layer|layer)" | head -30 || \
  info "init_test_all.py output suppressed — run manually for full report"

echo ""
echo -e "${BOLD}═══════════════════════════════════════════${NC}"
echo -e "${BOLD}  Setup complete.${NC}"
echo -e ""
echo -e "  To run the system:"
echo -e "    ${CYAN}python run_open.py${NC}    # Pre-market macro scan + universe ranking"
echo -e "    ${CYAN}python run_hourly.py${NC}  # Intraday signal updates"
echo -e "    ${CYAN}python run_close.py${NC}   # End-of-day portfolio reconciliation"
echo -e ""
echo -e "  HFT execution engine (standalone):"
echo -e "    ${CYAN}python src/agents/execution_engine.py${NC}"
echo -e ""
echo -e "  If API keys are missing, edit: ${CYAN}${ENV_FILE}${NC}"
echo -e "${BOLD}═══════════════════════════════════════════${NC}\n"

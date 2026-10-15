"""Study-wide settings. Everything that changes a result lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "perp.sqlite"
SQL_DIR = ROOT / "sql"
REPORT_DIR = ROOT / "reports"

ASSETS = ("BTC", "ETH", "SOL")
SYMBOLS = {a: f"{a}USDT" for a in ASSETS}

# Binance USDT-margined perpetuals. BTC's perp launched Sep 2019; SOL's in
# Sep 2020. Prices are USDT; USDT is treated as USD (see report caveats).
START_DATE = "2019-09-10"
END_DATE = "2026-10-14"      # last complete UTC day when the data was pulled

# Chronological split, as in the spot study. Thresholds and hedge rules are
# chosen on TRAIN; TEST is run once.
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"

DAYS_PER_YEAR = 365
# Costs per unit of notional traded, per leg. Taker fee + half-spread on a
# large venue; the cost grid in the report brackets this.
COST_BPS = 7.5
COST_GRID_BPS = (0, 5, 7.5, 15, 30)

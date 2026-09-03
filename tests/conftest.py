# ============================================================
# tests/conftest.py — Trading Brain
# ============================================================
# Rend les modules de la racine (ai_opinion, backtest_ranking, alerting)
# importables depuis tests/ sans installer le projet ni définir
# PYTHONPATH à la main.
#
# Lancement : depuis la racine du repo,  pytest -q
# ============================================================

import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RACINE not in sys.path:
    sys.path.insert(0, RACINE)

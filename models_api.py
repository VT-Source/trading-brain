# ============================================================
# models_api.py — Pydantic models pour l'API Trading Brain
# ============================================================

from pydantic import BaseModel
from typing import Optional


class DecisionPayload(BaseModel):
    semaine:     str            # format YYYY-MM-DD (lundi de la semaine)
    ticker:      str
    rang:        Optional[int]  = None
    decision:    str            # 'suivi' | 'ignore' | 'modifie'
    commentaire: Optional[str]  = None


class PositionOpenPayload(BaseModel):
    ticker:      str
    date_achat:  str              # format YYYY-MM-DD
    prix_achat:  float
    quantite:    float
    decision_id: Optional[int]  = None
    source:      Optional[str]  = "ranking"   # 'ranking' | 'manuel'
    commentaire: Optional[str]  = None


class PositionClosePayload(BaseModel):
    date_vente:   str            # format YYYY-MM-DD
    prix_vente:   float
    raison_vente: str            # TRAILING_STOP | TREND_BROKEN | MOMENTUM_LOST | SECTOR_WEAK | MACRO_BEARISH | MANUEL


class PositionEditPayload(BaseModel):
    prix_achat:  Optional[float] = None
    quantite:    Optional[float] = None
    date_achat:  Optional[str]   = None
    commentaire: Optional[str]   = None
    decision_id: Optional[int]   = None

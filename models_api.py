# ============================================================
# models_api.py — Pydantic models pour l'API Trading Brain
# v1.2 — ajout multi-portefeuille (PositionOpenPayload.portefeuille_id,
#        PortefeuilleCreatePayload, PortefeuilleEditPayload)
# v1.3 - devises (PositionOpen/Close : devise_saisie, montant_investi_eur)
# v1.4 - devises édition (PositionEditPayload : devise_saisie, montant_investi_eur)
# ============================================================

from pydantic import BaseModel
from typing import Optional


class PositionOpenPayload(BaseModel):
    ticker:          str
    date_achat:      str              # format YYYY-MM-DD
    prix_achat:      float
    quantite:        float
    source:          Optional[str] = "ranking"   # 'ranking' | 'manuel'
    commentaire:     Optional[str] = None
    portefeuille_id: Optional[int] = 1           # défaut : Philippe
    devise_saisie:       Optional[str]   = None   # ex 'USD' — None = prix déjà en devise de cotation
    montant_investi_eur: Optional[float] = None   # montant EUR réellement débité (avis d'opéré Saxo)


class PositionClosePayload(BaseModel):
    date_vente:   str            # format YYYY-MM-DD
    prix_vente:   float
    raison_vente: str            # TRAILING_STOP | TREND_BROKEN | MOMENTUM_LOST | SECTOR_WEAK
    devise_saisie: Optional[str] = None           # ex 'USD' — None = prix déjà en devise de cotation | MACRO_BEARISH | MANUEL


class PositionEditPayload(BaseModel):
    prix_achat:          Optional[float] = None
    quantite:            Optional[float] = None
    date_achat:          Optional[str]   = None
    commentaire:         Optional[str]   = None
    devise_saisie:       Optional[str]   = None   # ex 'USD' — convertit prix_achat vers la devise de cotation
    montant_investi_eur: Optional[float] = None   # EUR réellement débité (avis d'opéré Saxo)


class PortefeuilleCreatePayload(BaseModel):
    nom: str


class PortefeuilleEditPayload(BaseModel):
    nom:   Optional[str]  = None
    actif: Optional[bool] = None

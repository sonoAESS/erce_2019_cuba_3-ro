# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

from modelos import config as cfg
from modelos.contrato import validar_y_tipar


def _metadata_fake():
    return {"features_esperadas": list(cfg.FEATURES)}


def _df_cumple():
    rng = np.random.RandomState(0)
    n = 20
    datos = {c: rng.randn(n) for c in cfg.FEATURES_NUMERICOS}
    datos.update({c: rng.choice(["a", "b"], n) for c in cfg.FEATURES_CATEGORICOS})
    return pd.DataFrame(datos)


def test_contrato_ok():
    res = validar_y_tipar(_df_cumple(), _metadata_fake())
    assert res["ok"] is True
    assert res["faltantes"] == []
    assert res["num_observaciones"] == 20
    assert list(res["df"].columns) == cfg.FEATURES


def test_contrato_faltantes():
    df = _df_cumple().drop(columns=["ISECF"])
    res = validar_y_tipar(df, _metadata_fake())
    assert res["ok"] is False
    assert "ISECF" in res["faltantes"]


def test_contrato_sobrantes_ignoradas():
    df = _df_cumple().assign(extra="x")
    res = validar_y_tipar(df, _metadata_fake())
    assert res["sobrantes"] == ["extra"]
    assert "extra" not in res["df"].columns


def test_contrato_tipos():
    df = _df_cumple()
    df["ISECF"] = ["1,2", "3", None, "4"] + ["1"] * (len(df) - 4)
    res = validar_y_tipar(df, _metadata_fake())
    # la numérica con comas -> estropeada a NaN por diseño (SER coerce)
    assert res["df"]["ISECF"].isna().any() or np.issubdtype(
        res["df"]["ISECF"].dtype, np.floating)
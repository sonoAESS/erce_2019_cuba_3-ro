# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

from modelos import friedman


def _tabla_ruido():
    """3 algoritmos aleatorios sobre 6 folds."""
    rng = np.random.RandomState(0)
    return {"A": rng.rand(6), "B": rng.rand(6), "C": rng.rand(6)}


def test_ranking_final_campos():
    fr = friedman.ranking_final(_tabla_ruido())
    assert set(fr) >= {"friedman_stat", "friedman_p", "significativo",
                       "cd_nemenyi", "ranking", "nemenyi"}
    assert fr["n_folds"] == 6
    assert len(fr["ranking"]) == 3
    # rangos ordinales positivos
    assert (fr["ranking"]["rango_medio"] >= 1.0).all()


def test_tabla_por_algoritmo_ordena_por_fold():
    filas = pd.DataFrame({
        "fold": [0, 0, 1, 1, 2, 2],
        "algoritmo": ["X", "Y", "X", "Y", "X", "Y"],
        "metrica": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
    })
    t = friedman.tabla_por_algoritmo(filas)
    assert list(t) == ["X", "Y"]
    assert t["X"].tolist() == pytest.approx([0.9, 0.7, 0.5])


def test_diferencia_critica_valores():
    cd = friedman.diferencia_critica(10, 5)
    assert cd > 0
    assert friedman.diferencia_critica(0, 5) != cd  # n_folds 0 -> nan
    assert np.isnan(friedman.diferencia_critica(0, 5))
    assert np.isnan(friedman.diferencia_critica(10, 1))


def test_grafico_cd_genera_figura():
    fr = friedman.ranking_final(_tabla_ruido())
    fig = friedman.grafico_cd(fr["ranking"], fr["cd_nemenyi"])
    assert fig is not None
    assert len(fig.axes) == 1
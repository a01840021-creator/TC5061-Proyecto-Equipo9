import pandas as pd
import pytest

from mibici.features import generar_dataset_p2


@pytest.mark.parametrize("horizonte", [0, -1, 1.5, "2", True])
def test_p2_rechaza_horizonte_invalido(horizonte):
    viajes = pd.DataFrame(columns=["inicio_del_viaje", "fin_del_viaje", "origen_id", "destino_id"])
    estaciones = pd.DataFrame({"id": [1], "dpcapacity": [10]})
    with pytest.raises(ValueError, match="horizonte_horas"):
        generar_dataset_p2(viajes, estaciones, horizonte_horas=horizonte)


def test_p2_acepta_horizonte_de_una_hora():
    viajes = pd.DataFrame(
        {
            "inicio_del_viaje": pd.to_datetime(["2026-04-02 08:10:00"]),
            "fin_del_viaje": pd.to_datetime(["2026-04-02 08:40:00"]),
            "origen_id": [1],
            "destino_id": [2],
        }
    )
    estaciones = pd.DataFrame({"id": [1, 2], "dpcapacity": [10, 10]})
    df_p2 = generar_dataset_p2(viajes, estaciones, horizonte_horas=1)
    assert "saldo_futuro_1h" in df_p2.columns

"""Contratos de entrada y salida de la API de inferencia de MiBici."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


FEATURES_P1 = [
    "estacion_id",
    "dpcapacity",
    "hora",
    "dia_semana",
    "es_fin_de_semana",
    "rezago_24h",
    "rezago_48h",
    "rezago_168h",
]


class DemandaEntrada(BaseModel):
    """Variables utilizadas por el modelo champion de demanda horaria P1."""

    model_config = ConfigDict(extra="forbid", strict=True)

    estacion_id: int = Field(ge=1, description="Identificador numérico de la estación.")
    dpcapacity: float = Field(ge=0, description="Capacidad registrada de la estación.")
    hora: int = Field(ge=0, le=23, description="Hora del día en formato de 0 a 23.")
    dia_semana: int = Field(ge=0, le=6, description="Día de la semana: lunes=0, domingo=6.")
    es_fin_de_semana: int = Field(
        ge=0,
        le=1,
        description="Indicador binario: 1 para sábado o domingo y 0 en otro caso.",
    )
    rezago_24h: float = Field(ge=0, description="Demanda observada 24 horas antes.")
    rezago_48h: float = Field(ge=0, description="Demanda observada 48 horas antes.")
    rezago_168h: float = Field(ge=0, description="Demanda observada una semana antes.")

    @model_validator(mode="after")
    def validar_fin_de_semana(self):
        """Mantiene consistente el indicador con el día de la semana recibido."""
        esperado = int(self.dia_semana in {5, 6})
        if self.es_fin_de_semana != esperado:
            raise ValueError(
                "es_fin_de_semana no coincide con dia_semana "
                "(sábado=5 y domingo=6)."
            )
        return self


class DemandaSalida(BaseModel):
    """Predicción de demanda con los datos de linaje necesarios para auditarla."""

    model_config = ConfigDict(allow_inf_nan=False)

    prediction: float = Field(description="Demanda horaria estimada por el modelo.")
    model_name: str
    alias: str
    version: str
    model_uri: str
    latency_ms: float = Field(ge=0)


class HealthResponse(BaseModel):
    """Estado de salud del proceso o de disponibilidad del modelo."""

    status: str
    model_status: str


class ModelInfoResponse(BaseModel):
    """Metadatos del modelo recuperado desde MLflow Model Registry."""

    model_name: str | None
    alias: str | None
    version: str | None
    model_uri: str
    status: str

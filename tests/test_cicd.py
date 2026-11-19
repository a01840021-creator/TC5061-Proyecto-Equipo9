"""Pruebas de contrato para el pipeline de CI/CD, Docker Compose y Dockerfile."""

from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[1]
WORKFLOW_CI = RAIZ / ".github" / "workflows" / "ci.yml"
COMPOSE_FILE = RAIZ / "docker-compose.yml"
DOCKERFILE = RAIZ / "Dockerfile"


def _cargar_yaml(ruta: Path) -> dict:
    assert ruta.is_file(), f"No se encontró el archivo esperado: {ruta}"
    contenido = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    assert isinstance(contenido, dict)
    return contenido


def test_workflow_cicd_disparadores_y_jobs():
    """El workflow corre en push/pull_request a main y encadena verificación con contenedor."""
    workflow = _cargar_yaml(WORKFLOW_CI)

    # PyYAML (YAML 1.1) interpreta la clave sin comillas 'on' como True
    disparadores = workflow.get("on") or workflow.get(True)
    assert isinstance(disparadores, dict)
    assert "main" in disparadores["push"]["branches"]
    assert "main" in disparadores["pull_request"]["branches"]

    jobs = workflow["jobs"]
    assert "verificar" in jobs
    assert "construir-y-validar" in jobs
    assert jobs["construir-y-validar"]["needs"] == "verificar"


def test_workflow_verificar_mantiene_estilo_pruebas_y_seguridad():
    """El job de verificación ejecuta ruff, pytest y make security con caché de pip."""
    workflow = _cargar_yaml(WORKFLOW_CI)
    pasos = workflow["jobs"]["verificar"]["steps"]

    setup_python = [p for p in pasos if str(p.get("uses", "")).startswith("actions/setup-python")]
    assert len(setup_python) == 1
    assert setup_python[0]["with"]["cache"] == "pip"

    comandos = "\n".join(p.get("run", "") for p in pasos)
    assert "ruff check" in comandos
    assert "ruff format --check" in comandos
    assert "pytest" in comandos
    assert "make security" in comandos


def test_workflow_construir_y_validar_cubre_build_cache_compose_y_smoke():
    """El job de contenedores construye con caché GHA, levanta Compose, prueba y limpia."""
    workflow = _cargar_yaml(WORKFLOW_CI)
    pasos = workflow["jobs"]["construir-y-validar"]["steps"]

    uses = [str(p.get("uses", "")) for p in pasos]
    assert any(u.startswith("docker/setup-buildx-action") for u in uses)

    build_steps = [p for p in pasos if str(p.get("uses", "")).startswith("docker/build-push-action")]
    assert len(build_steps) == 1
    build_with = build_steps[0]["with"]
    assert build_with["load"] is True
    assert "type=gha" in str(build_with["cache-from"])
    assert "type=gha" in str(build_with["cache-to"])

    comandos = "\n".join(p.get("run", "") for p in pasos)
    assert "docker compose config" in comandos
    assert "docker compose up -d" in comandos
    assert "http://localhost:5001/health" in comandos
    assert "scripts/smoke_test.sh" in comandos

    limpieza = [p for p in pasos if "docker compose down" in p.get("run", "")]
    assert len(limpieza) == 1
    assert limpieza[0].get("if") == "always()"


def test_docker_compose_define_servicios_red_volumen_y_limites():
    """docker-compose.yml define API, MLflow, inicialización de permisos, red y volumen."""
    compose = _cargar_yaml(COMPOSE_FILE)

    servicios = compose["services"]
    assert {"api", "mlflow", "mlflow-init"}.issubset(servicios.keys())

    for nombre in ("api", "mlflow"):
        srv = servicios[nombre]
        assert "healthcheck" in srv
        assert "cpus" in srv
        assert "mem_limit" in srv
        assert "MIBICI_IMAGE" in srv.get("image", "")

    assert "mlflow-data" in compose["volumes"]
    assert "mibici-network" in compose["networks"]


def test_dockerfile_mantiene_usuario_no_root_y_healthcheck():
    """El Dockerfile corre como usuario sin privilegios y expone HEALTHCHECK."""
    texto = DOCKERFILE.read_text(encoding="utf-8")
    assert "USER api" in texto
    assert "HEALTHCHECK" in texto

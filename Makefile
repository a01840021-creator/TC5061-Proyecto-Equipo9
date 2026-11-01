#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = TC5061-Proyecto-Equipo9
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

# Riesgo aceptado en pip-audit. La versión corregida de pillow (13.0.0), la de tornado (6.6) y
# la de filelock (4.2.1) exigen Python 3.11 o superior y el proyecto fija Python 3.10. Cada
# identificador se ignora de forma explícita; el análisis y el control compensatorio están en
# references/seguridad.md
VULNS_ACEPTADAS_PILLOW = \
	PYSEC-2026-4326 PYSEC-2026-4327 PYSEC-2026-4328 PYSEC-2026-4329 PYSEC-2026-4330 \
	PYSEC-2026-4331 PYSEC-2026-4332 PYSEC-2026-4333 PYSEC-2026-4334 PYSEC-2026-4335 \
	PYSEC-2026-4336 PYSEC-2026-4337 PYSEC-2026-4338 PYSEC-2026-4339 PYSEC-2026-4340 \
	PYSEC-2026-4341 PYSEC-2026-4342 PYSEC-2026-4343 CVE-2026-96638 CVE-2026-96640 \
	CVE-2026-96778 CVE-2026-112390 CVE-2026-112391 CVE-2026-112393 CVE-2026-112395 \
	CVE-2026-112399 CVE-2026-112400 CVE-2026-112401 CVE-2026-112404 CVE-2026-112406 \
	CVE-2026-112891 CVE-2026-112892 CVE-2026-112893 CVE-2026-112894 CVE-2026-112895 \
	CVE-2026-112897 CVE-2026-112899 CVE-2026-112900 GHSA-78g4-xqwv-q3g8
VULNS_ACEPTADAS_TORNADO = CVE-2026-115927 CVE-2026-116577 CVE-2026-116578
VULNS_ACEPTADAS_FILELOCK = \
	CVE-2026-112906 CVE-2026-112907 CVE-2026-112908 CVE-2026-112909 CVE-2026-112910 \
	CVE-2026-112911 CVE-2026-112912 CVE-2026-112913 CVE-2026-112914 CVE-2026-112915 \
	CVE-2026-112934 CVE-2026-112935 CVE-2026-112937 CVE-2026-112938 CVE-2026-112939 \
	CVE-2026-112940 CVE-2026-112941
VULNS_ACEPTADAS = $(VULNS_ACEPTADAS_PILLOW) $(VULNS_ACEPTADAS_TORNADO) $(VULNS_ACEPTADAS_FILELOCK)
IGNORAR_VULNS = $(foreach id,$(VULNS_ACEPTADAS),--ignore-vuln $(id))

# requirements.txt fija versiones exactas, así que pip-audit lo lee sin resolver dependencias
# ni crear un entorno temporal. La línea "-e ." aparece como omitida porque es el propio proyecto.
PIP_AUDIT = pip-audit -r requirements.txt --no-deps --disable-pip --progress-spinner off

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest -q tests


## Run DeepChecks data-quality reports on deterministic samples
.PHONY: quality
quality:
	$(PYTHON_INTERPRETER) -m mibici.quality --pregunta all --max-filas 5000


## Run security analysis (Bandit SAST and pip-audit SCA over requirements.txt)
.PHONY: security
security:
	bandit -c .bandit -r mibici/ tests/
	$(PIP_AUDIT) $(IGNORAR_VULNS)


## Generate detailed security reports in reports/security/
# El reporte de pip-audit no aplica IGNORAR_VULNS, para que liste también el riesgo aceptado
.PHONY: security-report
security-report:
	mkdir -p reports/security
	bandit -c .bandit -r mibici/ tests/ -f txt -o reports/security/bandit_report.txt || true
	$(PIP_AUDIT) -f markdown -o reports/security/pip_audit_report.md || true
	@echo "Security reports generated in reports/security/"


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) -m mibici.dataset


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)

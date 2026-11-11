"""
Phase 6 — deployment files stay consistent with the code.

These cannot replace a real deploy (that needs Render / Neon / Upstash
accounts), but they catch the mistakes that would break one: a secret
committed in a config file, a wrong health-check path, a start script with
Windows line endings, an image that would ship training tools.
"""
import pytest

from app.config.settings import PROJECT_ROOT

yaml = pytest.importorskip("yaml")


@pytest.fixture(scope="module")
def blueprint() -> dict:
    return yaml.safe_load((PROJECT_ROOT / "render.yaml").read_text(encoding="utf-8"))


def services_by_name(blueprint: dict) -> dict:
    return {service["name"]: service for service in blueprint["services"]}


def test_blueprint_defines_the_api_and_the_demo(blueprint):
    services = services_by_name(blueprint)

    assert set(services) == {"smartprice-api", "smartprice-demo"}
    assert all(service["plan"] == "free" for service in services.values())


def test_api_health_check_points_at_the_real_endpoint(blueprint):
    assert services_by_name(blueprint)["smartprice-api"]["healthCheckPath"] == "/health"


def test_secrets_are_never_committed(blueprint):
    for service in blueprint["services"]:
        for var in service.get("envVars", []):
            if var["key"] in {"DATABASE_URL", "REDIS_URL", "SMARTPRICE_API_URL"}:
                assert var.get("sync") is False and "value" not in var, f"{service['name']}: {var['key']} must not be committed"


def test_demo_installs_only_its_own_requirements(blueprint):
    demo = services_by_name(blueprint)["smartprice-demo"]

    assert demo["buildCommand"] == "pip install -r demo/requirements.txt"
    assert "$PORT" in demo["startCommand"]


def test_start_script_uses_port_and_unix_line_endings():
    script = (PROJECT_ROOT / "scripts" / "start.sh").read_bytes()

    assert b"\r\n" not in script                          # CRLF would make Linux fail with "not found"
    assert b"alembic upgrade head" in script and b"app.database.seed" in script
    assert b'--port "${PORT:-8000}"' in script


def test_dockerfile_installs_runtime_requirements_only():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    run_lines = [line for line in dockerfile.splitlines() if line.startswith("RUN")]
    assert "RUN pip install -r requirements.txt" in run_lines
    assert not any("requirements-dev" in line for line in run_lines)   # no training/test tools in production
    assert "USER smartprice" in dockerfile                 # not root


def test_dockerignore_keeps_secrets_and_bulk_out_of_the_image():
    ignored = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8").split()

    for entry in (".env", ".venv/", "data/raw/", "mlflow.db", "tests/"):
        assert entry in ignored

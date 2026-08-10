"""Two ways the dashboard broke without the API being wrong.

**The error body was not JSON.** Endpoints catch the errors they expect and return 4xx.
Anything else fell through to Starlette's default 500, whose body is the bare string
"Internal Server Error" — and every fetch in `index.html` calls `.json()` on the response.
So a server-side fault reached the user as `Unexpected token 'I', "Internal S"... is not
valid JSON`: a message that says nothing about what failed and points at the wrong layer.

**A served model was invisible.** `MODEL_ORDER` in the dashboard is filtered against what
the API returned, so listing a model that no longer exists is harmless. The reverse is not:
`gradient_boost` shipped, served correctly, appeared in `/predict`, and never once rendered,
because nothing listed it. There was no error anywhere to notice.
"""

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.api as api
from models.predict import ENSEMBLE_TIERS, _TIER_FACTORIES

FRONTEND = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


@pytest.fixture
def client():
    """A client with the lifespan run, so the models are actually loaded."""
    with TestClient(api.app, raise_server_exceptions=False) as c:
        yield c


class _Boom:
    """Stands in for the predictor and fails the way a real bug would."""

    _models = {"dixon_coles": None}

    def predict(self, *args, **kwargs):
        raise KeyError("gradient_boost")


# --- the error body ------------------------------------------------------------------

def test_an_unexpected_error_still_returns_json(client, monkeypatch):
    monkeypatch.setattr(api, "_predictor", _Boom())
    response = client.get("/predict/Arsenal/Chelsea")

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    response.json()          # the regression: this used to raise


def test_the_error_body_names_what_went_wrong(client, monkeypatch):
    """`{"detail": "Internal Server Error"}` would satisfy the test above and still leave
    nobody any wiser."""
    monkeypatch.setattr(api, "_predictor", _Boom())
    detail = client.get("/predict/Arsenal/Chelsea").json()["detail"]

    assert "KeyError" in detail and "gradient_boost" in detail


def test_expected_failures_are_still_4xx_not_500(client):
    """The catch-all must not swallow the deliberate ones."""
    assert client.get("/predict/Arsenal/Arsenal").status_code == 400
    assert client.get("/predict/Not A Real Club/Chelsea").status_code == 404


# --- the dashboard's model list ------------------------------------------------------

def _listed_in_dashboard(const: str) -> set[str]:
    html = FRONTEND.read_text(encoding="utf-8")
    block = re.search(rf"const {const}\s*=\s*([\[{{].*?[\]}}]);", html, re.S)
    assert block, f"{const} not found in index.html"
    return set(re.findall(r"['\"]?(\w+)['\"]?\s*[:,\]]", block.group(1))) - {""}


@pytest.fixture
def shipped_models() -> set[str]:
    return {_TIER_FACTORIES[t]().name for t in ENSEMBLE_TIERS} | {"ensemble"}


@pytest.mark.parametrize("const", ["MODEL_ORDER", "MODEL_NAMES", "MODEL_COLORS"])
def test_the_dashboard_lists_every_model_that_ships(const, shipped_models):
    missing = shipped_models - _listed_in_dashboard(const)
    assert not missing, f"served but never rendered — {const} is missing {sorted(missing)}"


def test_the_api_serves_exactly_the_models_that_ship(client, shipped_models):
    assert set(client.get("/predict/Arsenal/Chelsea").json()["tiers"]) == shipped_models


# --- surviving a bad pickle ----------------------------------------------------------

def test_one_unreadable_pickle_does_not_take_down_the_others(tmp_path, matches):
    """What actually happened in production: `load_models` looped over every .pkl and let
    the first failure propagate, so a deployment served zero models and an empty team list
    while three of the four files were perfectly readable.

    The two pickles holding scikit-learn estimators are the fragile ones — an estimator
    pickled under one version is not guaranteed to unpickle under another, and the machine
    that trains is not the machine that serves.
    """
    from models.dixon_coles import DixonColesModel
    from models.predict import MatchPredictor

    DixonColesModel().fit(matches).save(tmp_path / "dixon_coles.pkl")
    (tmp_path / "corrupt.pkl").write_bytes(b"not a pickle at all")

    predictor = MatchPredictor.load_models(tmp_path, matches)

    assert "dixon_coles" in predictor._models, "a readable model was lost to a broken one"
    assert "corrupt.pkl" in predictor.load_failures_


def test_health_is_degraded_not_ok_when_nothing_loaded(client, monkeypatch):
    """`{"status": "ok"}` with zero models is the one answer a health check must never
    give — it is what let a broken deployment look fine from the outside."""
    monkeypatch.setattr(api, "_predictor", None)
    monkeypatch.setattr(api, "_teams", [])
    monkeypatch.setattr(api, "_load_error", "RuntimeError: boom")

    body = client.get("/api/health").json()
    assert body["status"] == "degraded"
    assert "boom" in body["error"]


def test_health_reports_a_partial_load(client, monkeypatch):
    """The production case: two of three models loaded, every request got a real answer,
    and health said "ok". Serving a subset is not health."""
    monkeypatch.setattr(api._predictor, "load_failures_",
                        {"gradient_boost.pkl": "AttributeError: nope"}, raising=False)
    body = client.get("/api/health").json()

    assert body["status"] == "degraded"
    assert body["model_load_failures"] == {"gradient_boost.pkl": "AttributeError: nope"}


def test_health_reports_the_versions_it_is_running(client):
    """The outage was a version mismatch between the machine that pickled the models and
    the machine that unpickled them, and nothing served said what either was running."""
    runtime = client.get("/api/health").json()["runtime"]

    assert set(runtime) == {"python", "scikit_learn", "numpy", "pandas"}
    assert all(re.match(r"^\d+\.\d+", v) for v in runtime.values())


def test_the_pinned_scikit_learn_matches_the_one_that_pickles_models():
    """requirements.txt bounds scikit-learn to a minor line. If the environment that
    trains drifts outside it, the artifacts it commits will not load where they are
    served — which is exactly what happened, silently."""
    import sklearn
    from packaging.requirements import Requirement

    line = next(
        l for l in (Path(__file__).resolve().parents[1] / "requirements.txt")
        .read_text(encoding="utf-8").splitlines()
        if l.strip().startswith("scikit-learn")
    )
    assert sklearn.__version__ in Requirement(line.strip()).specifier, (
        f"training on scikit-learn {sklearn.__version__}, which requirements.txt excludes "
        f"({line.strip()}) — models pickled here will not load in the deployed API"
    )

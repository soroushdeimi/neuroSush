"""HTTP contracts for the first compose form."""

from fastapi.testclient import TestClient
from studio.app import app

client = TestClient(app)


def spec(tau=20.0):
    return {
        "layers": {
            "cortex": {
                "groups": {
                    "E": {
                        "shape": 1,
                        "inhibitory": False,
                        "tags": [],
                        "behaviors": [
                            {
                                "kind": "LIF",
                                "params": {
                                    "tau": tau,
                                    "threshold": -50,
                                    "v_reset": -65,
                                    "v_rest": -65,
                                },
                            }
                        ],
                    }
                },
                "synapses": [],
                "inputs": None,
                "outputs": None,
            }
        },
        "synapses": [],
        "inputs": {},
        "outputs": {},
    }


def test_health_and_schema():
    assert client.get("/api/health").json() == {"status": "ok"}
    response = client.get("/api/schema")
    assert response.status_code == 200
    assert any(b["name"] == "LIF" for b in response.json()["behaviors"])


def test_validate_constructor_and_path():
    assert client.post("/api/validate", json=spec()).json() == {
        "valid": True,
        "errors": [],
    }
    result = client.post("/api/validate", json=spec(0)).json()
    assert result["valid"] is False
    assert result["errors"][0]["path"] == "/layers/cortex/groups/E/behaviors/0/params/tau"
    assert "0" in result["errors"][0]["message"]
    assert result["errors"][0]["fix"]


def test_malformed_and_unknown():
    assert client.post("/api/validate", json={}).json()["valid"] is False
    data = spec()
    data["layers"]["cortex"]["groups"]["E"]["behaviors"][0]["kind"] = "NotRegistered"
    error = client.post("/api/validate", json=data).json()["errors"][0]
    assert error["path"].endswith("/kind")
    assert "NotRegistered" in error["message"]


def test_synapse_errors_and_escaped_pointer():
    """Validate behaviors inside lists and escape group names in pointers."""
    data = spec()
    data["layers"]["cortex"]["synapses"] = [
        {
            "src": "E",
            "dst": "E",
            "compartment": "proximal",
            "behaviors": [{"kind": "UnknownSynapse", "params": {}}],
        }
    ]
    result = client.post("/api/validate", json=data).json()
    assert result["valid"] is False
    assert result["errors"][0]["path"] == "/layers/cortex/synapses/0/behaviors/0/kind"
    data = spec(0)
    data["layers"]["cortex"]["groups"]["E/~"] = data["layers"]["cortex"]["groups"].pop("E")
    error = client.post("/api/validate", json=data).json()["errors"][0]
    assert "/groups/E~1~0/" in error["path"]


def test_wrong_types_and_missing_required_parameter():
    """Reject string numbers, booleans, unknown arguments and absent required values."""
    for tau in ("20", True, None, {"bad": "value"}):
        result = client.post("/api/validate", json=spec(tau)).json()
        assert result["valid"] is False
        assert result["errors"][0]["path"].endswith("/params/tau")
    data = spec()
    params = data["layers"]["cortex"]["groups"]["E"]["behaviors"][0]["params"]
    del params["tau"]
    params["surprise"] = 7
    result = client.post("/api/validate", json=data).json()
    assert {e["path"].split("/")[-1] for e in result["errors"]} == {"tau", "surprise"}


def test_invalid_shapes_and_json_body():
    """Reject invalid dimensions and malformed HTTP JSON."""
    for shape in (0, -1, [1, 2], [1, 0, 3], True):
        data = spec()
        data["layers"]["cortex"]["groups"]["E"]["shape"] = shape
        assert client.post("/api/validate", json=data).json()["valid"] is False
    assert (
        client.post(
            "/api/validate",
            content="{bad",
            headers={"Content-Type": "application/json"},
        ).status_code
        == 422
    )


def test_group_name_can_equal_structural_field():
    """Treat group names as names even when they match a spec field."""
    data = spec()
    groups = data["layers"]["cortex"]["groups"]
    groups["behaviors"] = groups.pop("E")
    assert client.post("/api/validate", json=data).json() == {"valid": True, "errors": []}

"""Cheap checks for health, readiness and model discovery."""

import sys


def assert_model_listing(listing, model, context, input_modalities):
    assert listing["object"] == "list", listing
    entries = listing["data"]
    assert len(entries) == 1, listing  # The runner starts an isolated text server.
    entry = entries[0]
    assert entry["object"] == "model" and entry["id"] == model, entry
    assert entry["owned_by"] == "gufo" and entry["context_length"] == context, entry
    assert type(entry["created"]) is int and entry["created"] > 0, entry
    assert entry["architecture"]["input_modalities"] == input_modalities, entry


def check_discovery(client, model, checks, context, input_modalities):
    # Absolute URLs keep native probes outside the SDK's /v1 base path.
    import httpx

    for path, expected in (
        ("/health", {"status": "ok"}), ("/v1/health", {"status": "ok"}),
        ("/ready", {"status": "ready", "model": model}),
        ("/v1/ready", {"status": "ready", "model": model}),
    ):
        response = client.get(client.base_url.copy_with(path=path), cast_to=httpx.Response)
        result = {"status": response.status_code, "body": response.json()}
        name = "discovery_" + path.strip("/").replace("/", "_")
        checks[name] = result
        print(f"CHECK {name}", file=sys.stderr, flush=True)
        assert result == {"status": 200, "body": expected}, result

    response = client.models.with_raw_response.list()
    listing = response.parse().to_dict()
    checks["discovery_models"] = {"status": response.status_code, "body": listing}
    print("CHECK discovery_models", file=sys.stderr, flush=True)
    assert response.status_code == 200, response.status_code
    assert_model_listing(listing, model, context, input_modalities)

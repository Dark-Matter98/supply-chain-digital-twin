from inbound_twin.kpi import littles_law


def test_littles_law_plant_segment(default_run_30):
    """L from the state integral vs lambda*W from batch records (independent paths)."""
    ll = littles_law(default_run_30)
    assert ll["L"] > 0
    assert abs(ll["L"] - ll["lambda_W"]) / ll["L"] < 0.02

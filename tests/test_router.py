from maragent.core.router import SmartRouter
from maragent.schemas import PerceptionResult


def test_low_dental_routes_to_fast_unsupervised():
    router = SmartRouter(
        {
            "supervised": ["DICDNet", "OSCNet+"],
            "unsupervised": ["ADN", "calimar_gan"],
            "fast_supervised": "OSCNet+",
            "fast_unsupervised": "calimar_gan",
        }
    )
    decision = router.route(
        PerceptionResult(body_part="Head", implant_type="Dental Filling", severity="Low"),
        "case.png",
    )
    assert decision.route == "fast_restoration"
    assert decision.models_to_run == ["calimar_gan"]
    assert decision.is_dental is True


def test_high_general_routes_to_all_supervised():
    router = SmartRouter(
        {
            "supervised": ["DICDNet", "OSCNet+"],
            "unsupervised": ["ADN", "calimar_gan"],
            "fast_supervised": "OSCNet+",
            "fast_unsupervised": "calimar_gan",
        }
    )
    decision = router.route(
        PerceptionResult(body_part="Pelvis", implant_type="Hip Replacement", severity="High"),
        "case.png",
    )
    assert decision.route == "all_model_race"
    assert decision.models_to_run == ["DICDNet", "OSCNet+"]
    assert decision.is_dental is False

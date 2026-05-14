import pytest

from comfortflow.domain.comfort.entities import PMVScore, PPDScore, IndoorClimate
from comfortflow.domain.control.entities import HVACSetpoints, BuildingState
from comfortflow.domain.monitoring.entities import DriftScore


class TestPMVScore:
    def test_valid_range(self) -> None:
        assert PMVScore(-3.0).value == -3.0
        assert PMVScore(3.0).value == 3.0

    def test_rejects_invalid(self) -> None:
        with pytest.raises(ValueError):
            PMVScore(3.5)

    def test_comfortable(self) -> None:
        assert PMVScore(0.0).is_comfortable
        assert not PMVScore(1.0).is_comfortable


class TestPPDScore:
    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError):
            PPDScore(-1.0)


class TestHVACSetpoints:
    def test_heating_below_cooling(self) -> None:
        sp = HVACSetpoints(heating_celsius=21.0, cooling_celsius=24.0)
        assert sp.heating_celsius < sp.cooling_celsius

    def test_rejects_inverted(self) -> None:
        with pytest.raises(ValueError):
            HVACSetpoints(heating_celsius=25.0, cooling_celsius=21.0)


class TestDriftScore:
    def test_from_psi_drifted(self) -> None:
        ds = DriftScore.from_psi(0.3)
        assert ds.is_drifted

    def test_from_psi_not_drifted(self) -> None:
        ds = DriftScore.from_psi(0.1)
        assert not ds.is_drifted

import pytest

from comfortflow.domain.comfort.entities import IndoorClimate, OccupantState, PMVScore
from comfortflow.domain.comfort.pmv import calculate_pmv, calculate_ppd


def _typical_office_climate() -> IndoorClimate:
    return IndoorClimate(
        air_temperature_celsius=23.0,
        radiant_temperature_celsius=23.0,
        relative_humidity_percent=50.0,
        air_velocity_ms=0.1,
    )


def _typical_occupant() -> OccupantState:
    return OccupantState(clothing_insulation_clo=0.6, metabolic_rate_met=1.2)


class TestCalculatePMV:
    def test_neutral_conditions_near_zero(self) -> None:
        pmv = calculate_pmv(_typical_office_climate(), _typical_occupant())
        assert -0.5 <= pmv.value <= 0.5

    def test_hot_conditions_positive(self) -> None:
        hot = IndoorClimate(30.0, 30.0, 70.0, 0.1)
        pmv = calculate_pmv(hot, _typical_occupant())
        assert pmv.value > 1.0

    def test_cold_conditions_negative(self) -> None:
        cold = IndoorClimate(18.0, 18.0, 40.0, 0.1)
        pmv = calculate_pmv(cold, _typical_occupant())
        assert pmv.value < -0.5

    def test_warm_conditions_positive(self) -> None:
        warm = IndoorClimate(28.0, 28.0, 60.0, 0.1)
        pmv = calculate_pmv(warm, _typical_occupant())
        assert pmv.value > 0.5

    def test_extreme_clamped_to_range(self) -> None:
        extreme = IndoorClimate(40.0, 40.0, 90.0, 0.1)
        pmv = calculate_pmv(extreme, _typical_occupant())
        assert -3.0 <= pmv.value <= 3.0


class TestCalculatePPD:
    def test_neutral_pmv_gives_five_percent(self) -> None:
        ppd = calculate_ppd(PMVScore(0.0))
        assert ppd.value == 5.0

    def test_higher_pmv_gives_higher_ppd(self) -> None:
        ppd_low = calculate_ppd(PMVScore(0.5))
        ppd_high = calculate_ppd(PMVScore(2.0))
        assert ppd_high.value > ppd_low.value

    def test_ppd_symmetric(self) -> None:
        ppd_pos = calculate_ppd(PMVScore(1.0))
        ppd_neg = calculate_ppd(PMVScore(-1.0))
        assert ppd_pos.value == ppd_neg.value


class TestPMVScore:
    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            PMVScore(4.0)

    def test_comfortable_in_band(self) -> None:
        assert PMVScore(0.3).is_comfortable
        assert not PMVScore(1.5).is_comfortable

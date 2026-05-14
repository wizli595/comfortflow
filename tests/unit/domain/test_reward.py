from comfortflow.domain.control.reward import compute_reward


class TestComputeReward:
    def test_comfortable_gets_bonus(self) -> None:
        r = compute_reward(energy_kwh=1.0, pmv=0.3)
        assert r > compute_reward(energy_kwh=1.0, pmv=0.6)

    def test_more_energy_more_penalty(self) -> None:
        r_low = compute_reward(energy_kwh=0.5, pmv=0.0)
        r_high = compute_reward(energy_kwh=2.0, pmv=0.0)
        assert r_low > r_high

    def test_neutral_pmv_is_best(self) -> None:
        r = compute_reward(energy_kwh=1.0, pmv=0.0)
        assert r > compute_reward(energy_kwh=1.0, pmv=2.0)

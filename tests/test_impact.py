import unittest

from causal.data import make_campaign_data
from causal.impact import control_sensitivity, estimate_impact, placebo_test


class ImpactTests(unittest.TestCase):
    def test_recovers_true_lift(self):
        d = make_campaign_data(lift=0.08)
        r = estimate_impact(d.df, "test", d.controls, d.campaign_start, d.campaign_end, n_boot=200)
        true_cum = d.true_effect[d.campaign_start : d.campaign_end].sum()
        lo, hi = r.cumulative_interval
        self.assertLess(lo, true_cum)
        self.assertGreater(hi, true_cum)
        self.assertLess(r.pre_period_mape, 0.05)

    def test_no_effect_when_no_campaign(self):
        d = make_campaign_data(lift=0.0)
        r = estimate_impact(d.df, "test", d.controls, d.campaign_start, d.campaign_end, n_boot=200)
        lo, hi = r.relative_interval
        self.assertLess(lo, 0.0)
        self.assertGreater(hi, 0.0)

    def test_placebo_p_value_is_small_for_real_effect(self):
        d = make_campaign_data(lift=0.08)
        r = estimate_impact(d.df, "test", d.controls, d.campaign_start, d.campaign_end, n_boot=50)
        r.placebo_effects = placebo_test(d.df, "test", d.controls, d.campaign_start, 42)
        self.assertLess(r.placebo_p_value, 0.1)

    def test_weights_are_non_negative(self):
        d = make_campaign_data()
        r = estimate_impact(d.df, "test", d.controls, d.campaign_start, d.campaign_end, n_boot=10)
        self.assertTrue((r.weights >= 0).all())

    def test_sensitivity_flags_contaminated_control(self):
        d = make_campaign_data(spillover={4: 0.6})
        sens = control_sensitivity(d.df, "test", d.controls, d.campaign_start, d.campaign_end)
        shift = (sens["relative_lift"] - sens.loc["all controls", "relative_lift"]).abs()
        self.assertEqual(shift.idxmax(), "without control_4")


if __name__ == "__main__":
    unittest.main()

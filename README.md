# Did the campaign work?

Measuring the incremental effect of a regional marketing campaign with a synthetic control: build a counterfactual for the test region from regions that did not get the campaign, then compare.

The data is synthetic with a known true lift, so the estimate can be checked against the truth.

![Campaign impact](images/impact.png)

## Approach

1. **Counterfactual.** On the pre-campaign period, the test region is regressed on six control regions (an intercept plus non-negative weights, so the counterfactual is a blend of real regions rather than an arbitrary linear fit).
2. **Effect.** Actual minus counterfactual during the campaign, per day and cumulated.
3. **Uncertainty.** Pre-period residuals are resampled in weekly blocks to put a 90% interval around the counterfactual.
4. **Placebo test.** The same analysis runs on fake campaign dates inside the pre-period, where the true effect is zero. The real lift should stand clearly outside that distribution.

![Placebo test](images/placebo.png)

## Results

| | Estimated | Ground truth |
|---|---|---|
| Incremental sales | 34,809 (90%: 32,029 to 38,051) | 34,969 |
| Relative lift | +7.40% (90%: +6.77% to +8.15%) | +7.44% |

The pre-period fit is tight (MAPE ≈ 2%), and the real lift is larger than every one of the 22 placebo estimates, which all stay within ±1.1%. Details are in [results.md](results.md).

## Run it

```bash
pip install -r requirements.txt
python run_demo.py
python -m unittest discover -s tests -t .
```

`causal/impact.py` works on any daily DataFrame with a `date` column, one test series and several control series.

## When this works, and when it doesn't

- **Controls must not be touched by the campaign.** Spillover from national media or shared online targeting biases the counterfactual towards the treated region and shrinks the measured lift.
- **The pre-period must be long and stable.** If the test region diverges from its controls for other reasons (a new store, a competitor), the counterfactual breaks. Check the pre-period fit and the placebo distribution before trusting the number.
- **Intervals cover noise, not model choice.** The bootstrap reflects day-to-day noise around a fixed set of weights. For a decision, compare a few control sets and see whether the conclusion holds.

## License

MIT

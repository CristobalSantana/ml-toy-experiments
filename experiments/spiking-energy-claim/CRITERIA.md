# Pre-registration

Frozen 2026-09-05, before any network was built and before any number was
measured. Not edited afterwards. Deviations are recorded in `README.md`.

## The claim being tested

Spiking neural networks are promoted on energy, not accuracy. The argument is
that a spiking neuron which does not fire costs nothing, so the right unit of
cost is the **synaptic operation** - one accumulate per spike per outgoing
synapse - rather than the dense multiply-accumulate an ordinary network pays
whether or not anything interesting happened. Quoted savings are usually one
to two orders of magnitude.

There is a factor the headline usually omits. A spiking network runs for `T`
timesteps per prediction. Whatever sparsity buys has to beat that `T` before
anything is saved, and with the input encoding most implementations actually
use - repeating the analogue input at every timestep - **the first layer is
dense every single timestep**, so it costs `T` times what the dense network's
first layer costs.

**This measures both numbers on one task, at matched accuracy.**

## Task

Next-hour German solar capacity factor, from
[`datasets/de/opsd_solar`](../../datasets/de/opsd_solar/). Train 2015-2017,
validate 2018, test 2019, split by calendar year and never shuffled. Scored on
daylight hours only, since predicting zero at night is not forecasting. The
forecasting question itself is answered in
[`solar-forecast-skill`](../solar-forecast-skill/); here the forecast is only
a task on which two architectures can be matched.

## Models

- `persistence`, `smart_persistence` - references.
- `ridge` - the linear reference.
- `mlp` - dense feedforward, one hidden layer, T = 1.
- `snn_direct` - LIF hidden layer, non-spiking leaky readout, analogue input
  injected as a constant current at every timestep.
- `snn_rate` - the same network, with the input Bernoulli-encoded into spikes
  so the first layer is sparse as well.

The SNN and the MLP use the **same hidden width**, so the comparison is
between architectures rather than between capacities.

## Counting rules, fixed in advance

- **MLP**: `n_in * H + H * n_out` multiply-accumulates per prediction.
- **SNN, input layer**: dense coding costs `T * n_in * H`; rate coding costs
  `(input spikes) * H`.
- **SNN, hidden layer**: `(hidden spikes summed over T) * n_out`.
- **Total** is the sum. A MAC and a synaptic operation are counted as one
  operation each. That favours the SNN, since an accumulate is cheaper than a
  multiply-accumulate in hardware; the point is that the conclusion survives
  the favourable convention.

## Predictions

- **P1** The MLP beats smart persistence on the test year. Control: if it
  does not, the task is broken and nothing else means anything.
- **P2** SNN accuracy rises with `T` and comes within 2% relative RMSE of the
  MLP by `T = 16` under direct coding.
- **P3** At the smallest `T` where the SNN matches the MLP within 2% relative
  RMSE, the SNN performs **more** total operations than the MLP, not fewer.
- **P4** The hidden layer really is sparse - mean spike rate below 20% - so
  that a failure of P3 is caused by `T` and the dense input layer rather than
  by neurons firing constantly.
- **P5** Counting only the hidden-to-output synapses, and ignoring the input
  layer as energy comparisons often do, the SNN performs **fewer** operations
  than the MLP. The accounting convention, not the architecture, decides the
  answer.
- **P6** Rate coding needs a larger `T` than direct coding to reach the same
  accuracy.

Reported as medians over 3 seeds, never best cells.

## What would make this uninteresting

If the SNN never reaches MLP accuracy at any `T` tested, P3 and P5 are
unanswerable and the result is only "it did not work here". That gets
reported as such rather than dressed up.

# Gaussian HMM Report for AAPL

Period: 2013-01-01 to 2023-12-31

See generated CSVs, PNGs and tables in this outputs directory.

State summary:

|   state |         mean |       std |   occupancy |   occupancy_pct |
|--------:|-------------:|----------:|------------:|----------------:|
|       0 |  0.00154045  | 0.0117483 |        2082 |        0.753256 |
|       1 | -0.000894532 | 0.0283435 |         682 |        0.246744 |

Transition matrix:

|    |        S0 |        S1 |
|:---|----------:|----------:|
| S0 | 0.983662  | 0.0163383 |
| S1 | 0.0513196 | 0.94868   |

Forecast:

{
  "last_state": 0,
  "next_state_probs": {
    "0": 0.9836617011052379,
    "1": 0.016338298894762134
  }
}
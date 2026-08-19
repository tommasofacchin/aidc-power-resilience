# Step 2 — Preliminary Round

## Deadline

**24 August 2026, 23:59 CET**

Are you ready to tackle the challenge of the **Nuremberg Tech Arena**?

All Round One deliverables must be submitted by **midnight CET on 24 August 2026**.

---

## Required Deliverables

Each participant or team must submit the following materials.

### 1. Technical Report

Submit a report of **3–8 pages** covering:

- Data acquisition and preprocessing strategy:
  - Data-source selection
  - Missing-value treatment
  - Timestamp alignment
  - Spatial matching strategies
- Feature engineering methodology and its rationale
- Model architecture and training strategy
- Class-imbalance handling scheme
- Experimental results and ablation analysis
- Documentation of every data source used, including its associated license

### 2. Prediction Results File

For the designated test time window, submit a CSV file containing the prediction results.

The file must follow the formatting specifications defined in the problem statement.

### 3. Complete Codebase and Reproducibility Guide

Submit executable code for the entire workflow:

1. Data acquisition
2. Preprocessing
3. Feature construction
4. Model training
5. Prediction generation

Include an operational manual detailed enough for reviewers to reproduce the results from scratch. It should include:

- Environment configuration
- Data-acquisition scripts
- Required execution order
- Expected runtime estimates

---

# Topic 2 — AIDC Power Supply Resilience

## Hierarchical Prediction of Power Continuity Risk

### Resilience Prediction of AIDC Power Supply Under Extreme Scenarios

## Relevance

AI now runs on power as much as on chips. Large-scale GPU clusters operate continuously at extreme power densities, and even a few seconds of interruption can erase hours or days of training progress.

As a result, AIDC power-supply resilience is becoming a systemic risk across the AI industry chain.

However, this resilience is not fully controlled by the data center itself. Regardless of whether a facility adopts:

- UPS 2N
- UPS Distributed Redundancy (DR)
- HVDC 2N
- Direct Utility 2N

its ultimate power source is still the electrical grid.

Extreme weather can degrade regional distribution networks, particularly where overhead lines and radial topologies introduce vulnerable single points of failure. Once regional supply capacity is impaired, the risk propagates through incoming feeders to each AIDC site.

The final impact depends on the interaction between the external outage and the site's internal power architecture. Under the same weather and grid-risk conditions, different configurations of:

- Incoming lines
- UPS redundancy
- Diesel generators
- Energy storage

can lead to substantially different outage probabilities, critical-load coverage levels, and backup durations.

This is the core challenge: existing outage-prediction models usually stop at the regional level, estimating the probability, scope, and duration of grid failures. They rarely account for site-specific architecture, redundancy, or backup resources.

As a result, they cannot answer what operators truly need to know:

> Under this extreme-weather event, will my site stay up?  
> How much critical load can I protect?  
> For how long?

Closing this gap requires a hierarchical, site-level resilience prediction model.

Using weather, distribution-network risk, incoming-feeder conditions, and facility architecture as inputs, the model must predict:

- Power-supply risk score
- Critical-load coverage ratio
- Estimated backup duration

Predictions are required at two forecasting horizons:

| Forecast type | Horizon | Time resolution |
|---|---:|---:|
| Day-ahead forecasting | 48 hours | 1 hour |
| Hours-ahead forecasting | 6 hours | 5 minutes |

---

## Challenge 1 — Day-Ahead Power-Supply Risk Forecasting

The first challenge is to predict, one day ahead, whether a site will remain operational.

Given historical weather observations, regional power-supply risk values, and architecture configuration parameters, build a model that forecasts the **site-level power-supply risk score** for each of the following supply architectures:

- IT Power 2N + UPS 2N
- UPS Distributed Redundancy
- HVDC 2N
- Direct Utility 2N

Predictions must be generated at **1-hour resolution** over a **48-hour horizon**.

The risk score is defined as a sigmoidal transformation of the regional outage proportion. The parameters \(k\) and \(x_0\) are determined by the site's incoming-feeder topology:

- Single-feed
- Dual-feed, common source
- Dual-feed, independent source

The critical-load protection ratio and expected backup duration are then derived from the predicted score through rule-based functions.

The model must capture the interaction among:

- Weather conditions
- Regional network vulnerability
- Site architecture

It must also generalize across architectures, despite extreme outages being rare, high-impact events that can cause naïve models to predict low risk almost everywhere.

### Operational Value

Accurate day-ahead forecasting supports:

- **Advance contingency planning**, such as scheduling generator inspections, deferring non-critical workloads, and pre-positioning backup fuel before an extreme-weather event.
- **Portfolio-level risk management**, allowing operators to compare how much critical load each architecture can protect under the same forecast and informing site selection and capital allocation.

---

## Challenge 2 — Hours-Ahead Power-Supply Risk Forecasting

The second challenge is to predict, during the final hours before an event, exactly when and how severely a site will be affected.

Given the same inputs:

- Weather observations
- Regional risk values
- Architecture parameters

build a model that forecasts the site-level power-supply risk score at **5-minute resolution** over a **6-hour horizon**.

The model must strictly respect temporal causality as conditions evolve in near real time.

This finer granularity raises the difficulty: the model must track risk as it unfolds rather than simply predicting a coarse daily risk envelope. It must still correctly apply the topology-dependent sigmoid function and generalize across all four architectures.

### Operational Value

Accurate hours-ahead forecasting supports:

- **Real-time operations**, enabling load shedding or switchover to backup power at the correct moment: neither too early, which wastes capacity, nor too late, which risks load loss.
- **Situational awareness**, providing operators with a live and quantified estimate of remaining backup duration as an extreme event progresses.

---

## Data

Participants are provided with a curated list of recommended data sources across five dimensions, including API documentation and access guidance.

### Outage Data

- **NaFIRS LV Faults**  
  Primary source. Provides distribution-district-level data covering central-southern England and northern Scotland.

- **Eagle-I**  
  United States outage data, available as an optional reference source.

### Weather Data

- **Open-Meteo**  
  Participants should select one or more geographic coordinates with spatial granularity between **3 and 9 km**.

### Site Architecture Data

The organizing committee provides site power-supply architecture configuration parameters, including:

- Feeder topology
- Redundancy class
- Buffering characteristics

### Optional Data Sources

- **OpenStreetMap / Overpass API** for power-infrastructure information.
- **NOAA Storm Events Database** for extreme-weather event information.

## Evaluation

Evaluation will be performed using a similar dataset to the recommended sources described above.
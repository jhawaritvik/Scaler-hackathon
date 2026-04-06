# Fire Factor Validation

This document tracks which wildfire-environment interactions in the simulator
are directly grounded in published fire-behavior literature, which ones are
source-guided reductions, and which ones are still heuristic.

The goal is not to force a fake `15 x 15` fully dense coefficient matrix.
Wildfire behavior is better represented as a causal graph:

- weather influences fuel moisture and wind
- terrain shapes local temperature, humidity, and wind
- fuel moisture and fuel type control ignition resistance
- burning cells modify local heat, moisture, and airflow
- those local states then change spread and spotting

Trying to assign a direct scalar for every variable pair would be less
realistic than keeping the mechanistic intermediates explicit.

For the broader modeling philosophy behind that decision, see
[EMERGENCE_FIRST_MODELING_SPEC.md](./EMERGENCE_FIRST_MODELING_SPEC.md).

Also note:

- this document validates interaction factors, not every field in the codebase
- some fields are setup-only inputs or bookkeeping and should not be counted as
  modeled physical parameters

## Status Legend

- `Direct`: the formula or coefficient comes directly from a cited source.
- `Source-guided`: the structure comes from fire-science guidance, but the
  broad-category mapping or discretization is our modeling choice.
- `Heuristic`: physically motivated, but not yet calibrated to a specific
  published coefficient set for this reduced-order simulator.

## Canonical Sources

1. Rothermel, R. C. (1972), *A Mathematical Model for Predicting Fire Spread in
   Wildland Fuels*, USDA Forest Service Research Paper INT-115.
2. Alexandridis et al. CA spread model as reproduced in Freire and DaCamara
   (2019), *Using cellular automata to simulate wildfire propagation and to
   assist in fire management*, NHESS 19, 169-179.
3. Scott and Burgan (2005), *Standard Fire Behavior Fuel Models: A
   Comprehensive Set for Use with Rothermel's Surface Fire Spread Model*,
   USDA Forest Service RMRS-GTR-153.
4. NWCG S-190, *Principles of Wildland Fire Behavior*.
5. NASA POWER, *Relative Humidity* methodology note with saturation vapor
   pressure polynomial over water.
6. Albini (1983), *Potential spotting distance from wind-driven surface fires*,
   USDA Forest Service INT-RP-309.
7. Linn et al. (2025), *Wildland fire entrainment: The missing link between
   wildland fire and its environment*, PNAS.

## Validated Couplings

| From | To | Implementation | Status | Notes |
|---|---|---|---|---|
| wind speed + direction | spread probability | `exp(V * (c1 + c2 * (cos(theta) - 1)))` with `c1=0.045`, `c2=0.131`, `V` in m/s | `Direct` | Adopted from the Alexandridis CA family reproduced by Freire and DaCamara. |
| slope | spread probability | `exp(a_s * theta_s)` with `a_s=0.078` | `Direct` | Same Alexandridis formulation. `theta_s` is computed geometrically from elevation difference and cell distance. |
| fuel moisture | spread / ignition damping | Rothermel moisture damping polynomial | `Direct` | `eta_M = 1 - 2.59 r + 5.11 r^2 - 3.52 r^3`, `r = Mf / Mx`. |
| air temperature | local RH | local RH is recomputed from vapor pressure and saturation vapor pressure | `Direct` | Replaced the old linear RH-per-degree constant with a vapor-pressure calculation. |
| time lag | moisture equilibration rate | `alpha = 1 - exp(-dt/tau)` | `Direct` | Standard exponential response implied by dead-fuel timelag classes. |
| wind threshold | nonlocal spotting gate | `8 m/s` threshold | `Direct` | Taken from the modified Alexandridis-style wind propagation rule in Freire and DaCamara. |

## Source-Guided Couplings

| From | To | Implementation | Status | Notes |
|---|---|---|---|---|
| aspect | local temperature proxy | modest `[-1.0, +1.0] C` exposure offsets | `Source-guided` | NWCG states north slopes are cooler/wetter and south slopes hotter/drier; numeric discretization is ours. |
| aspect | initial fuel moisture | north-south spread of about `8` moisture points across extremes | `Source-guided` | Anchored to NWCG guidance that fuels in full sun can hold about `8%` less moisture than shade. |
| aspect | drying rate | `0.85x` to `1.15x` multiplier | `Source-guided` | Encodes the same north-vs-south solar exposure logic without overstating it. |
| fuel type | moisture timelag mapping | grass=`1h`, brush=`10h`, forest surface litter=`10h` | `Source-guided` | Timelag classes are standard; mapping our abstract fuel buckets onto them is a simplification. |
| fuel type | extinction moisture | grass=`0.15`, brush=`0.25`, forest=`0.30` | `Source-guided` | Representative values chosen from Scott and Burgan ranges for broad fuel groups. |
| elevation | local temperature | `0.65 C` per elevation level | `Source-guided` | Treated as a standard lapse rate over a `100 m` vertical level. |
| elevation | slope angle | `100 m` horizontal cell size, `100 m` per elevation level | `Source-guided` | Needed to convert synthetic terrain levels into geometric slopes. |
| roads / water / no-fuel | spread suppression | non-burnable or strongly damped cells | `Source-guided` | Consistent with Alexandridis-style `p_veg`/`p_den` reductions and NWCG barrier logic. |

## Remaining Heuristics

These parts are physically motivated and useful, but should not yet be treated
as empirically calibrated fire-science coefficients:

| Mechanism | Current Role | Why It Is Still Heuristic |
|---|---|---|
| fire-driven airflow potential | perturbs local wind through the gradient of a hidden potential field | This is a cleaner base-state replacement for the old updraft shortcut, but the decay, diffusion, and wind-scale coefficients are still reduced-order choices. |
| radiant fire drying strength | lowers nearby fuel moisture | Real mechanism, but our neighbor-radius weights are simulator-scale approximations. |
| adjacent-water humidity bonus | boosts local RH and moisture | Water barriers are real; the specific RH and recharge increments are not yet source-calibrated. |
| spotting probability curve | probability after the wind threshold gate | Albini supports spotting physics and long-range transport, but our compact probability law is still a gameplay-sized approximation. |
| burn duration and intensity curves | controls how long a cell burns and how intense it gets | Shape is sensible, but not yet fit to a published heat-release dataset for each abstract fuel class. |

Under the emergence-first design, several of these are not just "needs better
calibration" items. They are also candidates for removal if we can replace them
with cleaner base-state couplings.

## What Changed In Code

The simulator now uses the following stronger formulations:

1. Wind loading in spread uses the published Alexandridis exponential form,
   including the second directional coefficient `c2`, instead of a simplified
   single-coefficient cosine rule.
2. Wind speed is converted from `km/h` to `m/s` before applying the published
   wind formula.
3. Fuel moisture no longer attenuates ignition with `1 - moisture`; it now uses
   the Rothermel moisture damping polynomial.
4. Local RH is no longer reduced with a hand-tuned linear constant; it is
   derived from saturation vapor pressure under local heating.
5. Fuel moisture response rates are now derived from timelag classes and the
   simulation step duration rather than fixed ad hoc rates.
6. Aspect modifiers were reduced and re-centered so they match NWCG guidance
   more conservatively.
7. Spotting now requires stronger winds, aligned with the `8 m/s` gate used in
   the modified Alexandridis-style wind propagation rule.
8. The old crossover switch, gust flag, structure-targeted spotting, and
   valley/ridge wind shortcuts were removed from the live simulator.
9. Fire-atmosphere feedback now uses a hidden `airflow_potential` field,
   letting local flow perturbation come from a modeled state rather than from
   a direct special-case wind bonus.

## What This Means For Emergent Behavior

The model is now better positioned to create the kind of indirect behavior we
want:

- hotter cells lower local RH through vapor-pressure physics
- lower RH drives fuel moisture downward over time
- drier fuels receive a higher Rothermel moisture-damping coefficient
- stronger wind aligned with the spread direction amplifies ignition pressure
- uphill spread gains additional slope loading
- intense burning clusters can still bias local flow inward through the
  entrainment heuristic

That is still not a full CFD fire-atmosphere model, but it is much closer to
the right causal structure than a hand-written "if fire then extra wind" rule.

## Recommended Next Calibration Work

If we want to push realism further after the OpenEnv structure is complete, the
highest-value next steps are:

1. replace the spotting probability heuristic with a compact Albini-inspired
   transport model
2. tune burn duration and intensity against representative Scott-Burgan fuel
   models or BehavePlus outputs
3. replace heuristic terrain-wind factors with a compact channeling model based
   on local curvature / exposure
4. calibrate fire-induced inflow against reduced-order coupled-fire studies or
   FDS/WFDS-style benchmark cases

## Practical Bottom Line

We can now honestly separate the simulator into three layers:

- `well-anchored`: wind loading, slope loading, moisture damping, vapor-pressure
  RH response, timelag response
- `reasonable abstractions`: aspect scaling, broad fuel-category extinction
  moisture, terrain scale mapping
- `still heuristic`: water bonuses, fire-induced inflow strength, spotting
  probability curve, burn intensity curve

That is a much safer place to build the full environment on top of than the
previous mixture of correct formulas and undocumented constants.

# Synthetic Data Methodology

This document describes how ClimateGuard AI's synthetic Florida book of
business and simulated claims were generated, and — critically — is
explicit about which parts are grounded in real data/history versus
documented modeling assumptions. Every source module contains the same
methodology notes inline; this document consolidates them for review.

## 1. Synthetic Locations (Milestone 2.1)

**What's real:** the relative population weighting is anchored to real
Florida metro/county population figures (approximate, rounded — used only
as sampling weights, not presented as precise Census statistics).

**What's simplified:** rather than full US Census block-group shapefiles
(the roadmap's original plan), locations are sampled via Gaussian jitter
around 16 real metro population centers. This avoids a heavy geospatial
dependency chain for a hurricane-first vertical slice. See
`data_pipeline/synthetic/generate_locations.py` module docstring for the
full rationale.

**Result:** 20,000 locations, correctly weighted (Miami-Dade: 3,834
locations vs. Key West: 139 — roughly matching their real ~27:1 population
ratio).

## 2. Construction Attributes (Milestone 2.1)

**What's real:** the 2002 Florida Building Code reform (enacted after
Hurricane Andrew, 1992, exposed severe construction weaknesses) is a real,
verifiable regulatory event. It's used as a genuine breakpoint: locations
built after 2002 are modeled with a shift toward more wind-resistant
construction (masonry over frame, hip roofs over gable).

**What's simplified:** the *specific percentage splits* on either side of
that breakpoint (e.g. "55% masonry post-2002") are documented modeling
assumptions reflecting general domain knowledge, not digitized from a
specific cited statistical source.

## 3. Policies (Milestone 2.1)

**What's real:** Florida's percentage-of-dwelling-coverage hurricane
deductible convention (2%, 5%, or 10% of coverage, rather than a flat
dollar amount) is real, current Florida insurance-market practice (Fla.
Stat. § 627.701).

**What's simplified:** the specific weight distribution across
{2%, 5%, 10%} is a documented assumption.

## 4. Wind Field Model (Milestone 2.2)

**What's real:** the underlying physics — wind is strongest near a storm's
center and decays with distance, and more intense storms have tighter wind
fields — is correct and directionally grounded in real meteorology.

**What's simplified:** this is a symmetric radial power-law decay model,
not a literal reproduction of the Holland (1980) parametric wind field
model used in production cat models (which fits a pressure-based B-parameter
and models storm-motion asymmetry — the real right-front quadrant being
stronger than the rest of the storm). See
`data_pipeline/synthetic/wind_field_model.py` for full detail.

## 5. Vulnerability Curve (Milestone 2.2)

**What's real:** the S-shaped functional form (negligible damage below a
threshold, steep accumulation, saturating toward total loss) matches the
real shape used in published methodologies like FEMA HAZUS-HM. Construction-
class ordering (frame is least wind-resistant, reinforced concrete is most)
is well-established engineering knowledge.

**What's simplified:** the specific curve parameters (midpoint wind speeds,
steepness) are documented assumptions calibrated to be directionally
correct, not digitized from a specific published damage-ratio table.

## 6. Claims Generation & Validation Results (Milestone 2.2)

**Methodology:** the full historical HURDAT2 storm catalog (1851–2023,
filtered to 993 storms with track points near Florida) is applied to the
**current** synthetic book of business — standard catastrophe-modeling
practice (a historical event catalog represents a statistical sample of
possible events, applied to present-day exposure, not a literal historical
replay of what existed at each storm's actual date).

**Real, confirmed results** (run 2026-08-XX, seed=42):

| Metric | Value |
|---|---|
| Total claim records | 892,079 |
| Distinct storms producing a claim | 216 (of 993 filtered near Florida) |
| Total synthetic book TIV | $6,992,471,000 |
| Total simulated incurred loss (all storms, 1851–2023) | $41,603,935,953 |
| Implied average annual expected loss | ~3.5% of TIV/year |

**Top loss-driving storms** — independently corroborates the model, since
these are real, well-documented historically catastrophic Florida
hurricanes, recovered purely from feeding real track data through the
pipeline: the 1926 Miami hurricane, 1928 Okeechobee hurricane, 1935 Labor
Day hurricane (strongest US landfall on record), and Hurricane Andrew
(1992).

**Hurricane Ian (2022) sanity check:**

| Metric | Simulated | Real-world reference |
|---|---|---|
| Max wind experienced | 135 kt (~155 mph) | Ian's real landfall: ~150-155 mph (Cat 4, borderline 5) |
| Locations affected | 9,324 of 20,000 | Ian crossed much of the peninsula — plausible |
| Incurred loss | $537.2M (~7.7% of book TIV) | Directionally consistent with Ian being one of the costliest hurricanes in Florida history |

## Known limitations / what would need to change for production use

1. Wind field asymmetry (storm-motion effects) is not modeled.
2. Vulnerability curve parameters need calibration against a real cited
   damage-ratio table before loss ratios could be treated as
   representative of real market experience.
3. Storm-surge/flood damage is entirely out of scope — this is a
   wind-only model (`peril_type: "hurricane_wind"` throughout).
4. Location sampling uses metro-level, not block-group-level, population
   density.
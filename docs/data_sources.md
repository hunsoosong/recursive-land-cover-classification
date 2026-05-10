# External data sources

This experiment relies on two external datasets that are *not*
redistributed in this repository. They must be obtained directly from
their original providers.

## 1. Land-cover reference labels

**Lesiv et al. (2025).** Global land-cover validation reference dataset
(approximately 15 million human-interpreted point samples).

- Citation: Lesiv, M. et al. (2025). *Global land-cover validation
  reference dataset.* (See manuscript for full citation.)
- Availability: published with the original article.

The 13 original Lesiv categories are reduced to the 10-class scheme used in
this study by the mapping defined in `src/classes.py` (three minor or
non-target categories - burnt land, fallow, and 'not sure' - are excluded).

The raw reference table is expected to contain columns including the
sample latitude, longitude, and the original Lesiv class ID. After class
harmonisation, each surviving point carries an integer label in
`{0, ..., 9}` according to `src/classes.LABEL_TO_NAME`.

## 2. Geospatial foundation embeddings

**AlphaEarth Foundations** (Brown et al., 2025). 64-dimensional reusable
remote-sensing embedding at 10 m resolution, derived from Sentinel-1,
Sentinel-2, Landsat, and complementary sources.

- Citation: Brown, C. F. et al. (2025). *AlphaEarth Foundations.* (See
  manuscript for full citation.)
- Availability: Google Earth Engine asset
  `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`.

The Lesiv reference points are anchored to surface conditions in 2015,
whereas AlphaEarth coverage begins in 2017. We use the **2017 epoch**
throughout, as the closest available temporal match. Each reference point
is assigned the AlphaEarth embedding of the enclosing 10 m pixel.

## Notes on staging

The four-arm protocol in `src/protocol.py` consumes, for each sampling
seed, a held-out test set, a validation pool, and ten disjoint
class-balanced generation pools (`data1` ... `data10`). Each point
carries the AlphaEarth embedding of its enclosing 10 m pixel and an
integer label in `{0, ..., 9}` from the harmonised 10-class scheme. The
constants used to construct these splits (e.g. cell cap, per-class
target counts, scale-specific batch sizes) are exposed in
`src/config.py`; the procedure that produces them is described in
Section 2.3 of the manuscript. End-to-end re-runs from the original
data sources are possible but are not packaged as a one-line script in
this repository - see `docs/reproducibility.md` for the full scope.

# Data sources and licences

The files in this directory are derived data, reduced from two public
sources so that the example runs without network access. They are not
the original data sets.

- **ADS-B positions** (`counters.csv.gz`, `events.csv.gz`,
  `tracks-*.csv.gz`): derived from the adsb.lol global history archives
  (github.com/adsblol/globe_history_2025), © adsb.lol contributors, made
  available under the Open Database License 1.0 (ODbL), alternatively
  CC0-1.0. The derived files here are offered under the same ODbL terms:
  attribute adsb.lol, keep derived databases open. The aircraft
  identifiers are the 24-bit ICAO addresses as broadcast.
- **METAR reports** (`metar.csv.gz`): U.S. National Weather Service
  aviation weather observations, public domain, retrieved from the Iowa
  Environmental Mesonet ASOS archive (mesonet.agron.iastate.edu) of Iowa
  State University.

Live mode reads the adsb.lol API (api.adsb.lol) and the aviationweather.gov
data API directly; their terms apply to that use. Neither source is
affiliated with this project.

// Command airspace watches eight European airports the way an operator
// watches a network: aircraft movements are the counters, flight-weather
// reports the event stream. It trains on ordinary days, replays Storm
// Éowyn (2025-01-24) with the models running, and can run on the live
// feeds. See docs/manual/applications.md and testdata/ATTRIBUTION.md.
package main

import "math"

// airport is one watched field with its approach zone.
type airport struct {
	ICAO      string
	Name      string
	Lat, Lon  float64
	ElevFt    float64
	tzOffset  int // hours from UTC in January, for the day-of-week feature only
	radiusNM  float64
	ceilingFt float64
}

// airports are ordered west to east so a front is seen crossing them.
// (Belfast was the first choice for the north; the receiver coverage there
// in the archive was too thin to count movements, Edinburgh's is dense.)
var airports = []airport{
	{"EIDW", "Dublin", 53.4213, -6.2701, 242, 0, 40, 10000},
	{"EGPH", "Edinburgh", 55.9508, -3.3615, 136, 0, 40, 10000},
	{"EGCC", "Manchester", 53.3537, -2.2750, 257, 0, 40, 10000},
	{"EGLL", "London Heathrow", 51.4775, -0.4614, 83, 0, 40, 10000},
	{"EHAM", "Amsterdam", 52.3086, 4.7639, -11, 1, 40, 10000},
	{"EBBR", "Brussels", 50.9014, 4.4844, 184, 1, 40, 10000},
	{"LFPG", "Paris CDG", 49.0097, 2.5479, 392, 1, 40, 10000},
	{"EDDF", "Frankfurt", 50.0333, 8.5706, 364, 1, 40, 10000},
}

func airportIndex(icao string) int {
	for i, a := range airports {
		if a.ICAO == icao {
			return i
		}
	}
	return -1
}

const (
	earthRadiusNM = 3440.065
	deg           = math.Pi / 180
)

// distanceNM is the great-circle distance in nautical miles.
func distanceNM(lat1, lon1, lat2, lon2 float64) float64 {
	dlat := (lat2 - lat1) * deg
	dlon := (lon2 - lon1) * deg
	a := math.Sin(dlat/2)*math.Sin(dlat/2) + math.Cos(lat1*deg)*math.Cos(lat2*deg)*math.Sin(dlon/2)*math.Sin(dlon/2)
	return 2 * earthRadiusNM * math.Asin(math.Min(1, math.Sqrt(a)))
}

// localNM projects a position onto a flat east/north frame around the
// airport, in nautical miles: good enough within a 40 NM zone.
func (a airport) localNM(lat, lon float64) (dx, dy float64) {
	dy = (lat - a.Lat) * 60
	dx = (lon - a.Lon) * 60 * math.Cos(a.Lat*deg)
	return dx, dy
}

// inBox is a cheap pre-filter: a square of the zone radius in degrees.
func (a airport) inBox(lat, lon float64) bool {
	dlat := a.radiusNM / 60
	dlon := a.radiusNM / 60 / math.Cos(a.Lat*deg)
	return math.Abs(lat-a.Lat) <= dlat && math.Abs(lon-a.Lon) <= dlon
}

// headingDelta is the signed change from heading h1 to h2 in degrees,
// in (-180, 180].
func headingDelta(h1, h2 float64) float64 {
	d := math.Mod(h2-h1+540, 360) - 180
	if d <= -180 {
		d += 360
	}
	return d
}

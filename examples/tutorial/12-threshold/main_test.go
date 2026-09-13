package main

import (
	"sort"
	"testing"
)

// hourly is the first false-alarm reduction the chapter recommends, so
// it had better average correctly.
func TestHourlyAverages(t *testing.T) {
	minutes := make([]float32, 120)
	for i := range minutes {
		if i < 60 {
			minutes[i] = 2
		} else {
			minutes[i] = 5
		}
	}
	h := hourly(minutes)
	if len(h) != 2 {
		t.Fatalf("got %d hours from 120 minutes, want 2", len(h))
	}
	if h[0] != 2 || h[1] != 5 {
		t.Errorf("hours are %v, want [2 5]", h)
	}
}

func TestQuantile(t *testing.T) {
	s := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	if got := quantile(s, 0); got != 1 {
		t.Errorf("q=0 is %v, want 1", got)
	}
	if got := quantile(s, 1); got != 10 {
		t.Errorf("q=1 is %v, want 10", got)
	}
	if got := quantile(s, 0.5); got < 4 || got > 6 {
		t.Errorf("q=0.5 is %v, want the middle of the range", got)
	}
	if quantile(nil, 0.5) != 0 {
		t.Error("an empty sample should not panic")
	}
}

// The chapter's central claim is that the distributions overlap: the
// quiet day's worst hour exceeds every hour the model trained on. If
// that ever stops being true the chapter's argument changes, and the
// test should say so rather than the reader discovering it.
func TestQuietDayExceedsTrainingMaximum(t *testing.T) {
	train := []float32{4.1, 4.9, 5.5, 6.2, 6.92}
	quiet := []float32{4.0, 5.2, 7.82}
	sort.Slice(train, func(i, j int) bool { return train[i] < train[j] })
	sort.Slice(quiet, func(i, j int) bool { return quiet[i] < quiet[j] })
	if quantile(quiet, 1) <= quantile(train, 1) {
		t.Error("this fixture no longer demonstrates the overlap the chapter is about")
	}
}

func TestBtoi(t *testing.T) {
	if btoi(true) != 1 || btoi(false) != 0 {
		t.Error("btoi is wrong, which would silently corrupt every confusion matrix")
	}
}

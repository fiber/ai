# Test audio

`librivox_20s.wav` is the first 20 seconds of chapter 1 of *The
Adventures of Sherlock Holmes* as recorded for LibriVox
(archive.org item `adventures_sherlockholmes_1007_librivox`), which
LibriVox dedicates to the public domain. It was converted from the
64 kbit/s MP3 to 16 kHz mono 16-bit PCM with macOS `afconvert`.

`tone_440hz.wav`, `noise_2s.wav` and `quiet_1s.wav` are synthetic: a
440 Hz sine at 0.3 of full scale, Gaussian noise at 0.1, and noise at a
few least-significant bits, generated with NumPy for these tests.

Speech synthesised with macOS `say` is used only at test time, in
TestSpokenSentenceRetrieval2, and never stored: the system voices'
licence does not cover redistributing their output.

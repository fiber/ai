# Test images

All but two were drawn by a short Go program for these tests (shapes,
gradients, noise, stripes, an alpha ramp) and are released with the
repository under its licence.

`earth.jpg` is "The Earth seen from Apollo 17" (NASA, AS17-148-22727),
a work of the United States government in the public domain, fetched
from Wikimedia Commons at 500×500. `earth.png` is the same image
decoded by libjpeg (Pillow) and stored losslessly, so that the parity
test sees the pixels the reference sees; `earth.jpg` is decoded by Go's
decoder in the test that measures how much that difference matters.

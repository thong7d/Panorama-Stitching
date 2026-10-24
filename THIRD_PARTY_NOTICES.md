# Third-Party Notices and Algorithm References

Project code written by Group 19:

- Pham Minh Thong ? 23120364
- Nguyen Le Tan Tien ? 23120369
- Dao Quoc Tuan ? 23120392

## Incorporated third-party code/data

1. **OpenCV ORB sampling pattern**
   - The 256-entry `bit_pattern_31_` coordinate table in
     `src/v0_sequential.cpp` and `scripts/orb_pattern_data.py` is adapted
     from OpenCV's `modules/features2d/src/orb.cpp`.
   - Source: https://github.com/opencv/opencv/blob/4.x/modules/features2d/src/orb.cpp
   - License: BSD-3-Clause,
     https://github.com/opencv/opencv/blob/4.x/LICENSE

2. **stb image I/O**
   - `src/stb_image.h` and `src/stb_image_write.h` are unmodified vendored
     libraries by Sean Barrett and contributors.
   - Source: https://github.com/nothings/stb
   - License: public domain or MIT, at the user's choice, as stated upstream.

No other project implementation code was copied. Pillow, NumPy, OpenCV-Python,
Matplotlib, pandas, nbformat and CUDA are runtime/build dependencies used
through their documented APIs.

## Algorithm references

- E. Rosten and T. Drummond, ?Machine Learning for High-Speed Corner
  Detection,? ECCV 2006. https://doi.org/10.1007/11744023_34
- E. Rublee et al., ?ORB: An Efficient Alternative to SIFT or SURF,? ICCV
  2011. https://doi.org/10.1109/ICCV.2011.6126544
- D. Lowe, ?Distinctive Image Features from Scale-Invariant Keypoints,? IJCV
  2004. https://doi.org/10.1023/B:VISI.0000029664.99615.94
- M. Fischler and R. Bolles, ?Random Sample Consensus,? CACM 1981.
  https://doi.org/10.1145/358669.358692
- R. Hartley and A. Zisserman, *Multiple View Geometry in Computer Vision*,
  2nd ed., Cambridge University Press, 2004.

These publications inform the algorithm design; they are not sources of copied
implementation code.

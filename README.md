# a-sixel

A-Sixel library for encoding sixel images.

## Basic Usage

```rust
use a_sixel::BitMergeSixelEncoderBest;
use image::RgbImage;

let img = RgbImage::new(100, 100);
println!("{}", <BitMergeSixelEncoderBest>::encode(&img));
```

## Choosing an Encoder
- I want good quality:
  - Use `BitMergeSixelEncoderBest` or `KMeansSixelEncoder`.
- I'm time constrained:
  - Use `BitMergeSixelEncoderLow`, `BitSixelEncoder`, or `OctreeSixelEncoder`.
- I'm _really_ time constrained and can sacrifice a little quality:
  - Use `BitSixelEncoder<NoDither>`.

For a more detailed breakdown, here's the encoders by average speed and quality against the test
images (speed figures will vary) at 256 colors with Sierra dithering:

| Algorithm         |  MSE   | DSSIM  | Execution Time (ms) | Initial Buckets |
| :---------------- | :----: | :----: | ------------------: | --------------: |
| adu               | 15.56  | 0.0054 |                1473 |             N/A |
| bit               | 36.42  | 0.0132 |                 367 |             N/A |
| bit-merge-low     | 12.10  | 0.0046 |                 559 |            2^14 |
| bit-merge         | 10.96  | 0.0040 |                1198 |            2^18 |
| bit-merge-better  | 10.77  | 0.0039 |                2196 |            2^20 |
| bit-merge-best    | 10.75  | 0.0039 |                2850 |            2^21 |
| focal             | 11.72  | 0.0043 |                3253 |            2^21 |
| k-means           | 10.86  | 0.0040 |                6208 |             N/A |
| k-medians         | 18.68  | 0.0075 |               10688 |             N/A |
| median-cut        | 20.27  | 0.0061 |                 627 |             N/A |
| octree (max-heap) | 66.60  | 0.0163 |                 589 |             N/A |
| octree (min-heap) | 332.29 | 0.0890 |                 552 |             N/A |


Here's the encoders at 16 colors with Sierra dithering:
| Algorithm         |    MSE | DSSIM  | Execution Time (ms) | Initial Buckets |
| :---------------- | -----: | :----: | ------------------: | --------------: |
| adu               | 120.45 | 0.0371 |                 280 |             N/A |
| bit               | 182.07 | 0.0492 |                 247 |             N/A |
| bit-merge         |  98.10 | 0.0307 |                 993 |            2^18 |
| focal             | 107.17 | 0.0335 |                3045 |            2^21 |
| k-means           |  97.02 | 0.0297 |                 664 |             N/A |
| k-medians         | 171.20 | 0.0486 |                5792 |             N/A |
| median-cut        | 168.88 | 0.0381 |                 317 |             N/A |
| octree (max-heap) | 546.05 | 0.0922 |                 341 |             N/A |
| octree (min-heap) | 879.79 | 0.2536 |                 349 |             N/A |

License: MIT OR Apache-2.0
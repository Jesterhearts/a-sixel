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

| Algorithm        |   MSE | DSSIM  | PHash Distance | Mean ΔE | Max ΔE | ΔE >2.3 | ΔE >5.0 | Execution Time (ms) |
| :--------------- | ----: | :----: | -------------: | ------: | -----: | ------: | ------: | ------------------: |
| adu              | 15.56 | 0.0054 |           8.66 |    1.79 |  12.88 |     32% |    4.4% |                1473 |
| bit              | 36.42 | 0.0132 |          31.14 |    3.16 |  11.03 |     65% |   15.1% |                 367 |
| bit-merge-low    | 12.10 | 0.0046 |          13.97 |    1.95 |   9.98 |     32% |    2.2% |                 559 |
| bit-merge        | 10.96 | 0.0040 |          13.38 |    1.89 |  10.01 |     31% |    2.2% |                1198 |
| bit-merge-better | 10.77 | 0.0039 |          13.34 |    1.85 |  10.21 |     31% |    2.2% |                2196 |
| bit-merge-best   | 10.75 | 0.0039 |          13.58 |    1.85 |  10.21 |     31% |    2.2% |                2850 |
| focal            | 14.98 | 0.0057 |          20.21 |    2.30 |   9.16 |     45% |    3.3% |                2396 |
| k-means          | 10.86 | 0.0040 |          13.41 |    1.80 |  10.17 |     29% |    2.2% |                6208 |
| k-medians        | 18.68 | 0.0075 |          18.52 |    2.60 |  10.17 |     53% |    5.1% |               10688 |
| median-cut       | 20.27 | 0.0061 |          16.45 |    2.24 |  10.36 |     42% |    5.9% |                 627 |
| octree           | 66.60 | 0.0163 |          26.03 |    3.89 |  12.49 |     79% |   25.4% |                 589 |


Here's the encoders at 16 colors with Sierra dithering:

| Algorithm  |    MSE | DSSIM  | PHash Distance | Mean ΔE | Max ΔE | ΔE >2.3 | ΔE >5.0 | Execution Time (ms) |
| :--------- | -----: | :----: | -------------: | ------: | -----: | ------: | ------: | ------------------: |
| adu        | 120.45 | 0.0371 |          39.83 |    4.02 |  18.39 |     66% |     33% |                 280 |
| bit        | 182.07 | 0.0492 |          59.79 |    5.53 |  16.61 |     89% |     51% |                 247 |
| bit-merge  |  98.10 | 0.0307 |          41.79 |    3.96 |  16.41 |     68% |     31% |                 993 |
| focal      | 121.73 | 0.0362 |          48.97 |    4.36 |  16.90 |     72% |     34% |                2171 |
| k-means    |  97.02 | 0.0297 |          42.83 |    3.99 |  16.40 |     67% |     31% |                 664 |
| k-medians  | 171.20 | 0.0486 |          57.38 |    5.41 |  16.62 |     91% |     49% |                5792 |
| median-cut | 168.88 | 0.0381 |          45.28 |    4.68 |  16.72 |     74% |     42% |                 317 |
| octree     | 546.05 | 0.0922 |          75.07 |    7.69 |  18.87 |     98% |     74% |                 341 |



This doesn't tell the full story, as sometimes a low MSE or DSSIM can be achieved while losing some
highlight colors in the image. Take `flowers.png` for example:

<details> <summary>Flowers Base</summary>
<img src="test_images/flowers.png" />
</details>

<details> <summary>Flowers K-Means 16 Colors</summary>

This preserves the grey shades that make up the image well, but completely loses the blue of the
flowers at the base of the trees.

- MSE: 3.23
- DSSIM: 0.0020
- PHash Distance: 10
- Mean ΔE: 1.42
- Max ΔE: 11.49
- ΔE >2.3: 14%
- ΔE >5.0: 0%

<img src="example_images/flowers-k-means-16.png" />
</details>

<details> <summary>Flowers Focal 16 Colors</summary>

- MSE: 9.15
- DSSIM: 0.0091
- PHash Distance: 16
- Mean ΔE: 2.10
- Max ΔE: 9.23
- ΔE >2.3: 38%
- ΔE >5.0: 2% 

This sacrifices some differentiation between shades of grey, but preserves the blue of the flowers.

<img src="example_images/flowers-focal-16.png" />
</details>

License: MIT OR Apache-2.0
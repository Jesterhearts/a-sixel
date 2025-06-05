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

| Algorithm        |   MSE | DSSIM  | PHash Distance | Execution Time (ms) |
| :--------------- | ----: | :----: | -------------: | ------------------: |
| adu              | 15.56 | 0.0054 |           8.66 |                1473 |
| bit              | 36.42 | 0.0132 |          31.14 |                 367 |
| bit-merge-low    | 12.10 | 0.0046 |          13.97 |                 559 |
| bit-merge        | 10.96 | 0.0040 |          13.38 |                1198 |
| bit-merge-better | 10.77 | 0.0039 |          13.34 |                2196 |
| bit-merge-best   | 10.75 | 0.0039 |          13.58 |                2850 |
| focal            | 14.98 | 0.0057 |          20.21 |                2396 |
| k-means          | 10.86 | 0.0040 |          13.41 |                6208 |
| k-medians        | 18.68 | 0.0075 |          18.52 |               10688 |
| median-cut       | 20.27 | 0.0061 |          16.45 |                 627 |
| octree           | 66.60 | 0.0163 |          26.03 |                 589 |


Here's the encoders at 16 colors with Sierra dithering:
| Algorithm  |    MSE | DSSIM  | PHash Distance | Execution Time (ms) |
| :--------- | -----: | :----: | -------------: | ------------------: |
| adu        | 120.45 | 0.0371 |          39.83 |                 280 |
| bit        | 182.07 | 0.0492 |          59.79 |                 247 |
| bit-merge  |  98.10 | 0.0307 |          41.79 |                 993 |
| focal      | 121.73 | 0.0362 |          48.97 |                2171 |
| k-means    |  97.02 | 0.0297 |          42.83 |                 664 |
| k-medians  | 171.20 | 0.0486 |          57.38 |                5792 |
| median-cut | 168.88 | 0.0381 |          45.28 |                 317 |
| octree     | 546.05 | 0.0922 |          75.07 |                 341 |

This doesn't tell the full story, as sometimes a low MSE or DSSIM can be achieved while losing some
highlight colors in the image. Take `flowers.png` for example:

<details> <summary>Flowers Base</summary>
<img src="test_images/flowers.png" />
</details>

<details> <summary>Flowers K-Means 16 Colors</summary>
MSE: 3.23, DSSIM: 0.0020, PHash Distance: 10

This preserves the grey shades that make up the image well, but completely loses the blue of the
flowers at the base of the trees.

<img src="example_images/flowers-k-means-16.png" />
</details>

<details> <summary>Flowers Focal 16 Colors</summary>
MSE: 9.15, DSSIM: 0.0091, PHash Distance: 10

This sacrifices some differentiation between shades of grey, but preserves the blue of the flowers.

<img src="example_images/flowers-focal-16.png" />
</details>

License: MIT OR Apache-2.0
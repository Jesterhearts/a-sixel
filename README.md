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

| Algorithm         |    MSE | DSSIM  | Execution Time (ms) | Initial Buckets |
| :---------------- | -----: | :----: | ------------------: | --------------: |
| adu               |  15.56 | 0.0054 |                1473 |             N/A |
| bit               |  36.42 | 0.0132 |                 367 |             N/A |
| bit-merge-low     |  12.10 | 0.0046 |                 559 |            2^14 |
| bit-merge         |  10.96 | 0.0040 |                1198 |            2^18 |
| bit-merge-better  |  10.77 | 0.0039 |                2196 |            2^20 |
| bit-merge-best    |  10.75 | 0.0039 |                2850 |            2^21 |
| focal             |  14.98 | 0.0057 |                2396 |             N/A |
| k-means           |  10.86 | 0.0040 |                6208 |             N/A |
| k-medians         |  18.68 | 0.0075 |               10688 |             N/A |
| median-cut        |  20.27 | 0.0061 |                 627 |             N/A |
| octree (max-heap) |  66.60 | 0.0163 |                 589 |             N/A |
| octree (min-heap) | 332.29 | 0.0890 |                 552 |             N/A |


Here's the encoders at 16 colors with Sierra dithering:
| Algorithm         |    MSE | DSSIM  | Execution Time (ms) | Initial Buckets |
| :---------------- | -----: | :----: | ------------------: | --------------: |
| adu               | 120.45 | 0.0371 |                 280 |             N/A |
| bit               | 182.07 | 0.0492 |                 247 |             N/A |
| bit-merge         |  98.10 | 0.0307 |                 993 |            2^18 |
| focal             | 121.73 | 0.0362 |                2171 |             N/A |
| k-means           |  97.02 | 0.0297 |                 664 |             N/A |
| k-medians         | 171.20 | 0.0486 |                5792 |             N/A |
| median-cut        | 168.88 | 0.0381 |                 317 |             N/A |
| octree (max-heap) | 546.05 | 0.0922 |                 341 |             N/A |
| octree (min-heap) | 879.79 | 0.2536 |                 349 |             N/A |

This doesn't tell the full story, as sometimes a low MSE or DSSIM can be achieved while losing some
highlight colors in the image. Take `flowers.png` for example:

<details> <summary>Flowers Base</summary>
<img src="test_images/flowers.png" />
</details>

<details> <summary>Flowers K-Means 16 Colors</summary>
MSE: 3.23, DSSIM: 0.0020

This preserves the grey shades that make up the image well, but completely loses the blue of the
flowers at the base of the trees.

<img src="example_images/flowers-k-means-16.png" />
</details>

<details> <summary>Flowers Focal 16 Colors</summary>
MSE: 9.15, DSSIM: 0.0091

This sacrifices some differentiation between shades of grey, but preserves the blue of the flowers.

<img src="example_images/flowers-focal-16.png" />
</details>

License: MIT OR Apache-2.0
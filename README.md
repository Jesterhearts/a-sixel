# a-sixel

A sixel library for encoding images.

## Basic Usage

```rust
use a_sixel::BitMergeSixelEncoderBest;
use image::RgbaImage;

let img = RgbaImage::new(100, 100);
println!("{}", <BitMergeSixelEncoderBest>::encode(img));
```

## Transparency
By default, `a-sixel` handles transparency by setting any fully-transparent pixels to all-bits-zero.
This translates to a transparent pixel in most sixel implementations, but some terminals may not
support this.

Sixel does not natively support partial transparency, but this library does have some support for
rendering images as if partial transparency was supported. If the `partial-transparency` feature is
enabled, `a-sixel` will query the terminal and attempt to determine the background color. Partially
transparent pixels will then be blended with this background color before encoding. Note that with
this approach, changing the terminal background color will not update partially transparent pixels
to match. You will need to re-encode the image if the background color changes.


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
| adu              | 15.07 | 0.0053 |           8.66 |    1.79 |  12.88 |   31.6% |    4.4% |                1416 |
| bit              | 35.82 | 0.0132 |          31.14 |    3.16 |  11.03 |   64.5% |   15.1% |                 426 |
| bit-no-dither    | 31.78 | 0.0214 |          39.10 |    3.09 |  10.23 |   64.0% |   13.4% |                 274 |
| bit-merge-low    | 10.67 | 0.0038 |          13.97 |    1.95 |   9.98 |   32.4% |    2.2% |                 785 |
| bit-merge        | 10.36 | 0.0037 |          13.55 |    1.89 |  10.01 |   31.0% |    2.2% |                 932 |
| bit-merge-better | 10.31 | 0.0037 |          13.45 |    1.85 |  10.21 |   30.6% |    2.2% |                1275 |
| bit-merge-best   | 10.29 | 0.0037 |          13.52 |    1.83 |  10.26 |   30.6% |    2.2% |                1496 |
| focal            | 14.62 | 0.0056 |          19.97 |    2.30 |   9.16 |   45.3% |    3.3% |                2428 |
| k-means          | 10.07 | 0.0036 |          13.07 |    1.80 |  10.17 |   29.1% |    2.2% |                2996 |
| k-medians        | 17.67 | 0.0068 |          21.07 |    2.61 |  10.17 |   53.6% |    5.1% |                7305 |
| median-cut       | 19.63 | 0.0059 |          16.45 |    2.24 |  10.36 |   42.2% |    5.9% |                 692 |
| octree           | 54.48 | 0.0148 |          26.03 |    3.89 |  12.49 |   78.6% |   25.4% |                 682 |
| wu               | 17.89 | 0.0068 |          21.03 |    2.34 |  10.24 |   46.3% |    5.1% |                1853 |

**Note:** Execution time _includes_ the time taken to compute error statistics - this is
non-trivial. For example, exclusive of error statistics computation, bit-no-dither takes <100ms on
average. Performance figures will vary based on machine, etc. They are only useful for comparing
algorithms against each other within this dataset.

Here's the encoders at 16 colors with Sierra dithering:

| Algorithm  |    MSE | DSSIM | PHash Distance | Mean ΔE | Max ΔE | ΔE >2.3 | ΔE >5.0 | Execution Time (ms) |
| :--------- | -----: | :---: | -------------: | ------: | -----: | ------: | ------: | ------------------: |
| adu        | 116.85 | 0.036 |          39.83 |    4.02 |  18.39 |     66% |     33% |                 332 |
| bit        | 178.47 | 0.049 |          59.79 |    5.53 |  16.61 |     89% |     51% |                 307 |
| bit-merge  |  95.17 | 0.030 |          41.52 |    3.95 |  16.16 |     67% |     31% |                 712 |
| focal      | 118.57 | 0.035 |          48.59 |    4.36 |  16.88 |     72% |     34% |                2150 |
| k-means    |  99.36 | 0.031 |          43.10 |    3.99 |  16.41 |     67% |     31% |                 637 |
| k-medians  | 166.88 | 0.050 |          60.59 |    5.48 |  16.77 |     88% |     52% |                5447 |
| median-cut | 164.52 | 0.037 |          45.28 |    4.68 |  16.72 |     74% |     42% |                 374 |
| octree     | 459.37 | 0.085 |          75.07 |    7.69 |  18.89 |     98% |     74% |                 446 |
| wu         | 125.84 | 0.039 |          50.52 |    4.48 |  16.70 |     75% |     39% |                 906 |



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
# a-sixel

[![Crates.io](https://img.shields.io/crates/v/a-sixel.svg?style=for-the-badge)](https://crates.io/crates/a-sixel)
[![Dependency status](https://deps.rs/repo/github/Jesterhearts/a-sixel/status.svg?style=for-the-badge)](https://deps.rs/repo/github/Jesterhearts/a-sixel)
[![Documentation](https://img.shields.io/docsrs/a-sixel/latest?style=for-the-badge)](https://docs.rs/a-sixel)

A sixel library for encoding images.

## Basic Usage

### Simple Encoding

```rust
use a_sixel::BitMergeSixelEncoderBest;
use image::RgbaImage;

let img = RgbaImage::new(100, 100);
println!("{}", <BitMergeSixelEncoderBest>::encode(img));
```

### Loading and Encoding an Image File

```rust
use a_sixel::KMeansSixelEncoder;
use image;

// Load an image from file
let image = image::open("examples/transparent.png").unwrap().to_rgba8();

// Encode with default settings (256 colors, Sierra dithering)
let sixel_output = <KMeansSixelEncoder>::encode(image);
println!("{}", sixel_output);
```

### Custom Palette Size and Dithering

```rust
use a_sixel::{BitSixelEncoder, dither::NoDither};

let image = image::open("examples/transparent.png").unwrap().to_rgba8();

// Use 16 colors with no dithering for faster encoding
let sixel_output = BitSixelEncoder::<NoDither>::encode_with_palette_size(image, 16);
println!("{}", sixel_output);
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

| Algorithm        |   MSE |  DSSIM | PHash Distance | Mean ΔE | Max ΔE | ΔE >2.3 | ΔE >5.0 | Execution Time (ms) |
| :--------------- | ----: | -----: | -------------: | ------: | -----: | ------: | ------: | ------------------: |
| adu              | 15.04 | 0.0052 |           8.86 |    1.79 |  12.80 |   31.8% |    4.4% |                1448 |
| bit              | 35.82 | 0.0132 |          31.14 |    3.16 |  11.03 |   64.5% |   15.1% |                 468 |
| bit-merge-low    | 10.67 | 0.0038 |          13.97 |    1.95 |   9.98 |   32.4% |    2.2% |                 855 |
| bit-merge        | 10.37 | 0.0037 |          13.48 |    1.89 |  10.03 |   31.0% |    2.2% |                1034 |
| bit-merge-better | 10.30 | 0.0037 |          13.07 |    1.85 |  10.22 |   30.6% |    2.2% |                1301 |
| bit-merge-best   | 10.28 | 0.0037 |          13.59 |    1.83 |  10.20 |   30.5% |    2.2% |                1532 |
| focal            | 14.65 | 0.0056 |          20.10 |    2.30 |   9.17 |   45.3% |    3.3% |                2538 |
| k-means          | 10.07 | 0.0036 |          13.28 |    1.80 |  10.14 |   29.1% |    2.2% |                3175 |
| k-medians        | 17.22 | 0.0067 |          19.10 |    2.56 |   9.98 |   50.8% |    4.7% |                9088 |
| median-cut       | 19.64 | 0.0059 |          16.45 |    2.24 |  10.36 |   42.2% |    5.9% |                 740 |
| octree           | 54.48 | 0.0148 |          26.03 |    3.89 |  12.49 |   78.6% |   25.4% |                 754 |
| wu               | 17.89 | 0.0068 |          21.03 |    2.34 |  10.24 |   46.3% |    5.1% |                1984 |

**Note:** Execution time _includes_ the time taken to compute error statistics - this is
non-trivial. For example, exclusive of error statistics computation, bit-no-dither takes <100ms on
average. Performance figures will vary based on machine, etc. They are only useful for comparing
algorithms against each other within this dataset.

Here's the encoders at 16 colors with Sierra dithering:

| Algorithm        |    MSE |  DSSIM | PHash Distance | Mean ΔE | Max ΔE | ΔE >2.3 | ΔE >5.0 | Execution Time (ms) |
| :--------------- | -----: | -----: | -------------: | ------: | -----: | ------: | ------: | ------------------: |
| adu              | 118.90 | 0.0364 |          40.86 |    4.04 |  18.38 |   65.7% |   33.8% |                 357 |
| bit              | 178.47 | 0.0490 |          59.79 |    5.53 |  16.61 |   89.0% |   51.4% |                 325 |
| bit-merge-low    |  95.61 | 0.0302 |          41.59 |    3.97 |  16.26 |   67.4% |   31.4% |                 631 |
| bit-merge        |  94.53 | 0.0302 |          41.48 |    3.95 |  16.15 |   67.0% |   31.3% |                 800 |
| bit-merge-better |  96.11 | 0.0299 |          41.55 |    3.96 |  16.17 |   67.9% |   31.4% |                1078 |
| bit-merge-best   |  95.44 | 0.0297 |          41.69 |    3.96 |  16.46 |   67.0% |   31.4% |                1297 |
| focal            | 116.27 | 0.0350 |          48.48 |    4.35 |  16.87 |   71.7% |   34.1% |                2313 |
| k-means          |  99.36 | 0.0309 |          42.83 |    3.99 |  16.39 |   66.9% |   31.4% |                 702 |
| k-medians        | 173.95 | 0.0533 |          59.62 |    5.57 |  16.23 |   90.8% |   49.7% |                7255 |
| median-cut       | 164.52 | 0.0374 |          45.28 |    4.68 |  16.72 |   73.7% |   42.3% |                 395 |
| octree           | 459.37 | 0.0845 |          75.03 |    7.69 |  18.87 |   98.3% |   73.5% |                 477 |
| wu               | 125.84 | 0.0386 |          50.52 |    4.48 |  16.70 |   74.5% |   39.2% |                 929 |

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
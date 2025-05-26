# a-sixel

A-Sixel library for encoding sixel images.

#### Basic Usage

```rust
use a_sixel::ADUSixelEncoder;
use image::RgbImage;

let img = RgbImage::new(100, 100);
println!("{}", <ADUSixelEncoder>::encode(&img));
```

### Choosing an Encoder
- I want fast encoding with good quality:
  - Use `KMeansSixelEncoder` or `ADUSixelEncoder`.
- I'm time constrained:
  - Use `ADUSixelEncoder`, `BitSixelEncoder`, or `OctreeSixelEncoder`. You can customize `ADU` by
    lowering the `STEPS` parameter to run faster if necessary while still getting good results.
- I'm _really_ time constrained and can sacrifice a little quality:
  - Use `BitSixelEncoder<NoDither>`.
- I want high quality encoding, and don't mind a bit more computation:
  - Use `FocalSixelEncoder`.
  - This matters a lot less if you're not crunching the palette down below 256 colors.
  - Note that this an experimental encoder. It will *likely* produce better comparable or better
    results than other encoders, but may not always do so. On the test images, for my personal
    preferences, I think it's slightly better - particularly at small palette sizes. It works by
    computing weights for each pixel based on saliancy maps and measures of local statistics. These
    weighted pixels are then fed into a weighted k-means algorithm to produce a palette.

License: MIT OR Apache-2.0
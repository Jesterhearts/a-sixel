# a-sixel

A-Sixel library for encoding sixel images.

#### Basic Usage

```rust
use a_sixel::KMeansSixelEncoder;
use image::RgbImage;

let img = RgbImage::new(100, 100);
println!("{}", <KMeansSixelEncoder>::encode(&img));
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
  - Use `KMediansSixelEncoder`.
  - This matters a lot less if you're not crunching the palette down below 256 colors.

License: MIT OR Apache-2.0
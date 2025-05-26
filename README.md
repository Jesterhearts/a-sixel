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
  - Use `ADUSixelEncoder`
- I'm time constrained:
  - Use `ADUSixelEncoder`, `BitSixelEncoder`, or `OctreeSixelEncoder`. You can customize `ADU` by
    lowering the `STEPS` parameter to run faster if necessary while still getting good results.
- I'm _really_ time constrained and can sacrifice a little quality:
  - Use `BitSixelEncoder<NoDither>`.
- I want high quality encoding, and don't mind a bit more computation:
  - Use `FocalSixelEncoder`.
  - This matters a lot less if you're not crunching the palette down below 256 colors.
  - Note that this an experimental encoder. It will *likely* produce better results than just
    `ADUSixelEncoder`, but it may not always do so. On the test images, for my personal preferences,
    I think it's slightly better - particularly at small palette sizes.

    <details>
    <summary>How it works</summary>

    Under the hood, it is a modified version of the `ADUSixelEncoder` that uses a weighted selection
    algorithm for its sample pixels. These weights are determined based on saliency maps and
    measures of statistical noise in the image.

    In addition to the weighted selection, the distance metric used to determine which cluster to
    place a pixel into also incorporates the weight. Similar pixels with different weights will be
    nudged towards clusters with similar weights. This is a mild effect, but it seems to improve
    things over basic clustering when there are a lot of similar colors in an image.

    </details>

License: MIT OR Apache-2.0
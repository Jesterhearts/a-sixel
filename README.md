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
  - Use `KMeansSixelEncoder`, `BitMergeSixelEncoderBetter`, or `ADUSixelEncoder`.
    `KMeansSixelEncoder` scores slightly better on metrics such as MSE and DSSIM while being ~3x
    slower than `ADUSixelEncoder`, which is a good compromise between speed and quality.
- I'm time constrained:
  - Use `ADUSixelEncoder`, `BitMergeSixelEncoder`, or `BitSixelEncoder`. You can customize `ADU` by
    lowering the `STEPS` parameter to run faster if necessary while still getting good results.
    `BitSixelEncoder` is surprisingly good for all its simplicity while being the fastest encoder by
    a substantial margin. `BitMergeSixelEncoder` is fractionally slower than `BitSixelEncoder` while
    being signficantly better on error metrics.
- I'm _really_ time constrained and can sacrifice a little quality:
  - Use `BitSixelEncoder<NoDither>`.
- I want high quality encoding, and don't mind a bit more computation:
  - Use `KMediansSixelEncoder` or `FocalSixelEncoder`. These can be _very_ slow, but they produce
    high quality results. On metrics such as MSE and DSSIM k-means is roughly equivalent while being
    ~2x faster, but for my personal preferences, I like the results of these encoders better.
  - This matters a lot less if you're not crunching the palette down below 256 colors. At 256
    colors, all algorithms will produce decent results.

For a more detailed breakdown, here's the top encoders by average speed and quality against the test
images (speed figures will vary by machine):

| Algorithm        |  MSE  | DSSIM  | Execution Time (ms) |
| :--------------- | :---: | :----: | ------------------: |
| bit              | 34.06 | 0.0124 |                 332 |
| bit-merge        | 16.30 | 0.0059 |                 492 |
| bit-merge-better | 14.07 | 0.0053 |                 732 |
| bit-merge-best   | 12.17 | 0.0043 |                2120 |
| adu              | 15.56 | 0.0054 |                1407 |
| k-means          | 10.86 | 0.0040 |                6273 |


License: MIT OR Apache-2.0
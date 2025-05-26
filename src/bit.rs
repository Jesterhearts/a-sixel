//! Encodes a palette by bucketing bit ranges into a power of two number of
//! buckets. This is very fast and produces ok results for most images at larger
//! palette sizes (e.g. 256).

use std::collections::HashSet;

use dilate::DilateExpand;
use ordered_float::OrderedFloat;
use palette::{
    IntoColor,
    Lab,
    Srgb,
};

use crate::{
    dither::Sierra,
    private,
    PaletteBuilder,
    SixelEncoder,
};

pub type BitSixelEncoderMono<D = Sierra> = SixelEncoder<BitPaletteBuilder<2>, D>;
pub type BitSixelEncoder4<D = Sierra> = SixelEncoder<BitPaletteBuilder<4>, D>;
pub type BitSixelEncoder8<D = Sierra> = SixelEncoder<BitPaletteBuilder<8>, D>;
pub type BitSixelEncoder16<D = Sierra> = SixelEncoder<BitPaletteBuilder<16>, D>;
pub type BitSixelEncoder32<D = Sierra> = SixelEncoder<BitPaletteBuilder<32>, D>;
pub type BitSixelEncoder64<D = Sierra> = SixelEncoder<BitPaletteBuilder<64>, D>;
pub type BitSixelEncoder128<D = Sierra> = SixelEncoder<BitPaletteBuilder<128>, D>;
pub type BitSixelEncoder256<D = Sierra> = SixelEncoder<BitPaletteBuilder<256>, D>;

#[derive(Debug, Clone, Copy)]
struct Bucket {
    color: (u64, u64, u64),
    count: usize,
}

#[derive(Debug)]
pub struct BitPaletteBuilder<const PALETTE_SIZE: usize> {
    buckets: Vec<Bucket>,
}

impl<const PALETTE_SIZE: usize> BitPaletteBuilder<PALETTE_SIZE> {
    const PALETTE_DEPTH: usize = PALETTE_SIZE.ilog2() as usize;
    const SHIFT: usize = 24 - Self::PALETTE_DEPTH;

    fn new() -> Self {
        BitPaletteBuilder {
            buckets: vec![
                Bucket {
                    color: (0, 0, 0),
                    count: 0,
                };
                PALETTE_SIZE
            ],
        }
    }

    fn insert(&mut self, color: Srgb<u8>) {
        let index = {
            let r = color.red.dilate_expand::<3>().value();
            let g = color.green.dilate_expand::<3>().value();
            let b = color.blue.dilate_expand::<3>().value();

            // Since elements to the right will get shifted off first, we put them in grb
            // order (order of most-least significant for luminance). This probably doesn't
            // make a huge difference, but the theory is nice.
            let rgb = g << 2 | r << 1 | b;

            (rgb >> Self::SHIFT) as usize
        };

        let node = &mut self.buckets[index];
        node.color.0 += color.red as u64;
        node.color.1 += color.green as u64;
        node.color.2 += color.blue as u64;
        node.count += 1;
    }
}

impl<const PALETTE_SIZE: usize> private::Sealed for BitPaletteBuilder<PALETTE_SIZE> {}
impl<const PALETTE_SIZE: usize> PaletteBuilder for BitPaletteBuilder<PALETTE_SIZE> {
    const PALETTE_SIZE: usize = PALETTE_SIZE;

    fn build_palette(image: &image::RgbImage) -> Vec<Lab> {
        let mut builder = Self::new();

        for pixel in image.pixels() {
            builder.insert(Srgb::<u8>::new(pixel[0], pixel[1], pixel[2]));
        }

        builder
            .buckets
            .into_iter()
            .filter(|node| node.count > 0)
            .map(|node| {
                let rgb = Srgb::new(
                    (node.color.0 / node.count as u64) as u8,
                    (node.color.1 / node.count as u64) as u8,
                    (node.color.2 / node.count as u64) as u8,
                );
                let lab: Lab = rgb.into_format().into_color();
                [
                    OrderedFloat(lab.l),
                    OrderedFloat(lab.a),
                    OrderedFloat(lab.b),
                ]
            })
            .collect::<HashSet<_>>()
            .into_iter()
            .map(|[l, a, b]| Lab::new(*l, *a, *b))
            .collect::<Vec<_>>()
    }
}

//! Encodes a palette by bucketing bit ranges into a power of two number of
//! buckets. This is very fast and produces ok results for most images at larger
//! palette sizes (e.g. 256).

use std::{
    collections::HashSet,
    sync::atomic::{
        AtomicU64,
        Ordering,
    },
};

use dilate::DilateExpand;
use ordered_float::OrderedFloat;
use palette::{
    IntoColor,
    Lab,
    Srgb,
};
use rayon::iter::ParallelIterator;

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

#[derive(Debug)]
pub(crate) struct Bucket {
    pub(crate) color: (AtomicU64, AtomicU64, AtomicU64),
    pub(crate) count: AtomicU64,
}

#[derive(Debug)]
pub struct BitPaletteBuilder<const PALETTE_SIZE: usize> {
    pub(crate) buckets: Vec<Bucket>,
}

impl<const PALETTE_SIZE: usize> BitPaletteBuilder<PALETTE_SIZE> {
    const PALETTE_DEPTH: usize = PALETTE_SIZE.ilog2() as usize;
    const SHIFT: usize = 24 - Self::PALETTE_DEPTH;

    pub(crate) fn new() -> Self {
        BitPaletteBuilder {
            buckets: Vec::from_iter(
                std::iter::repeat_with(|| Bucket {
                    color: (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)),
                    count: AtomicU64::new(0),
                })
                .take(PALETTE_SIZE),
            ),
        }
    }

    pub(crate) fn insert(&self, color: Srgb<u8>) {
        let index = Self::index(color);
        let node = &self.buckets[index];
        node.color.0.fetch_add(color.red as u64, Ordering::Relaxed);
        node.color
            .1
            .fetch_add(color.green as u64, Ordering::Relaxed);
        node.color.2.fetch_add(color.blue as u64, Ordering::Relaxed);
        node.count.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn index(color: Srgb<u8>) -> usize {
        let r = color.red.dilate_expand::<3>().value();
        let g = color.green.dilate_expand::<3>().value();
        let b = color.blue.dilate_expand::<3>().value();

        // Since elements to the right will get shifted off first, we put them in grb
        // order (order of most-least significant for luminance). This probably doesn't
        // make a huge difference, but the theory is nice.
        let rgb = g << 2 | r << 1 | b;

        (rgb >> Self::SHIFT) as usize
    }
}

impl<const PALETTE_SIZE: usize> private::Sealed for BitPaletteBuilder<PALETTE_SIZE> {}
impl<const PALETTE_SIZE: usize> PaletteBuilder for BitPaletteBuilder<PALETTE_SIZE> {
    const NAME: &'static str = "Bit";
    const PALETTE_SIZE: usize = PALETTE_SIZE;

    fn build_palette(image: &image::RgbImage) -> Vec<Lab> {
        let builder = Self::new();

        image.par_pixels().for_each(|pixel| {
            builder.insert(Srgb::<u8>::new(pixel[0], pixel[1], pixel[2]));
        });

        builder
            .buckets
            .into_iter()
            .filter(|node| node.count.load(Ordering::Relaxed) > 0)
            .map(|node| {
                let rgb = Srgb::new(
                    (node.color.0.load(Ordering::Relaxed) / node.count.load(Ordering::Relaxed))
                        as u8,
                    (node.color.1.load(Ordering::Relaxed) / node.count.load(Ordering::Relaxed))
                        as u8,
                    (node.color.2.load(Ordering::Relaxed) / node.count.load(Ordering::Relaxed))
                        as u8,
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

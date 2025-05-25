use std::collections::HashSet;

use dilate::DilateExpand;
use ordered_float::NotNan;
use palette::{
    IntoColor,
    Lab,
    Srgb,
};

use crate::{
    PaletteBuilder,
    private,
};

#[derive(Debug, Clone, Copy)]
struct Node {
    color: (u64, u64, u64),
    count: usize,
}

#[derive(Debug)]
pub struct OctreePaletteBuilder<const PALETTE_DEPTH: usize> {
    octree: Vec<Node>,
}

impl<const PALETTE_SIZE: usize> OctreePaletteBuilder<PALETTE_SIZE> {
    const PALETTE_DEPTH: usize = PALETTE_SIZE.ilog2() as usize;
    const SHIFT: usize = 24 - Self::PALETTE_DEPTH;

    fn new() -> Self {
        OctreePaletteBuilder {
            octree: vec![
                Node {
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

        let node = &mut self.octree[index];
        node.color.0 += color.red as u64;
        node.color.1 += color.green as u64;
        node.color.2 += color.blue as u64;
        node.count += 1;
    }
}

impl<const PALETTE_SIZE: usize> private::Sealed for OctreePaletteBuilder<PALETTE_SIZE> {}
impl<const PALETTE_SIZE: usize> PaletteBuilder for OctreePaletteBuilder<PALETTE_SIZE> {
    const PALETTE_SIZE: usize = PALETTE_SIZE;

    fn build_palette(image: &image::RgbImage) -> Vec<Lab> {
        let mut builder = Self::new();

        for pixel in image.pixels() {
            builder.insert(Srgb::<u8>::new(pixel[0], pixel[1], pixel[2]));
        }

        builder
            .octree
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
                    NotNan::new(lab.l).unwrap(),
                    NotNan::new(lab.a).unwrap(),
                    NotNan::new(lab.b).unwrap(),
                ]
            })
            .collect::<HashSet<_>>()
            .into_iter()
            .map(|[l, a, b]| Lab::new(*l, *a, *b))
            .collect::<Vec<_>>()
    }
}

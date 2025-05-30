//! Uses an octree to build a palette from an image.
//!
//! This is fast (although not as fast as the
//! [`BitPaletteBuilder`](crate::BitPaletteBuilder)) and produces decent
//! results. By default the currently implemented algorithm uses a max-heap for
//! merging nodes by pixel count, leading to the palette being built from the
//! most dominant colors in the image first, with less dominate colors being
//! folded in later. This seems to produce a more visually appealing palette
//! than a min-heap on the test images, although it is counterintuitive.
//!
//! You can reverse this behavior by setting the `USE_MIN_HEAP` type parameter
//! to `true`, which will use a min-heap instead.

use std::collections::{
    BinaryHeap,
    HashSet,
};

use dilate::DilateExpand;
use ordered_float::OrderedFloat;
use palette::{
    IntoColor,
    Lab,
    Srgb,
};
use rayon::iter::{
    IndexedParallelIterator,
    IntoParallelIterator,
    IntoParallelRefIterator,
    ParallelIterator,
};

use crate::{
    dither::Sierra,
    private,
    PaletteBuilder,
    SixelEncoder,
};

pub type OctreeSixelEncoderMono<D = Sierra, const USE_MIN_HEAP: bool = false> =
    SixelEncoder<OctreePaletteBuilder<2, USE_MIN_HEAP>, D>;
pub type OctreeSixelEncoder4<D = Sierra, const USE_MIN_HEAP: bool = false> =
    SixelEncoder<OctreePaletteBuilder<4, USE_MIN_HEAP>, D>;
pub type OctreeSixelEncoder8<D = Sierra, const USE_MIN_HEAP: bool = false> =
    SixelEncoder<OctreePaletteBuilder<8, USE_MIN_HEAP>, D>;
pub type OctreeSixelEncoder16<D = Sierra, const USE_MIN_HEAP: bool = false> =
    SixelEncoder<OctreePaletteBuilder<16, USE_MIN_HEAP>, D>;
pub type OctreeSixelEncoder32<D = Sierra, const USE_MIN_HEAP: bool = false> =
    SixelEncoder<OctreePaletteBuilder<32, USE_MIN_HEAP>, D>;
pub type OctreeSixelEncoder64<D = Sierra, const USE_MIN_HEAP: bool = false> =
    SixelEncoder<OctreePaletteBuilder<64, USE_MIN_HEAP>, D>;
pub type OctreeSixelEncoder128<D = Sierra, const USE_MIN_HEAP: bool = false> =
    SixelEncoder<OctreePaletteBuilder<128, USE_MIN_HEAP>, D>;
pub type OctreeSixelEncoder256<D = Sierra, const USE_MIN_HEAP: bool = false> =
    SixelEncoder<OctreePaletteBuilder<256, USE_MIN_HEAP>, D>;

#[derive(Debug, Clone, Copy)]
struct Node {
    children: [Option<usize>; 8],
    rgb: (u64, u64, u64),
    count: u32,
    parent: usize,
    depth: usize,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct Candidate<const MIN: bool> {
    index: usize,
    count: u32,
}

impl<const MIN: bool> PartialOrd for Candidate<MIN> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<const MIN: bool> Ord for Candidate<MIN> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if MIN {
            other.count.cmp(&self.count)
        } else {
            self.count.cmp(&other.count)
        }
    }
}

#[derive(Debug)]
pub struct OctreePaletteBuilder<const PALETTE_SIZE: usize, const USE_MIN_HEAP: bool = false> {
    nodes: Vec<Node>,
}

impl<const PALETTE_SIZE: usize, const USE_MIN_HEAP: bool>
    OctreePaletteBuilder<PALETTE_SIZE, USE_MIN_HEAP>
{
    fn new() -> Self {
        OctreePaletteBuilder {
            nodes: vec![Node {
                children: [None; 8],
                rgb: (0, 0, 0),
                count: 0,
                parent: usize::MAX,
                depth: 0,
            }],
        }
    }

    fn insert(&mut self, color: Srgb<u8>) {
        let r = color.red.dilate_expand::<3>().value();
        let g = color.green.dilate_expand::<3>().value();
        let b = color.blue.dilate_expand::<3>().value();

        // Bit order here doesn't really matter because we always take 3 bits at a time.
        // So we just use the same order as bit.rs
        let mut index_bits = g << 2 | r << 1 | b;
        let mut node_index = 0;

        for depth in 0..7 {
            let child_index = (index_bits & 0b111) as usize;
            index_bits >>= 3;

            let parent = node_index;

            if self.nodes[node_index].children[child_index].is_none() {
                self.nodes[node_index].children[child_index] = Some(self.nodes.len());
                self.nodes.push(Node {
                    children: [None; 8],
                    rgb: (0, 0, 0),
                    count: 0,
                    parent,
                    depth: depth + 1,
                });
            }

            node_index = self.nodes[node_index].children[child_index].unwrap();
        }

        let node = &mut self.nodes[node_index];
        node.rgb.0 += color.red as u64;
        node.rgb.1 += color.green as u64;
        node.rgb.2 += color.blue as u64;
        node.count += 1;
    }
}

impl<const PALETTE_SIZE: usize, const USE_MIN_HEAP: bool> private::Sealed
    for OctreePaletteBuilder<PALETTE_SIZE, USE_MIN_HEAP>
{
}
impl<const PALETTE_SIZE: usize, const USE_MIN_HEAP: bool> PaletteBuilder
    for OctreePaletteBuilder<PALETTE_SIZE, USE_MIN_HEAP>
{
    const NAME: &'static str = "Octree";
    const PALETTE_SIZE: usize = PALETTE_SIZE;

    fn build_palette(image: &image::RgbImage) -> Vec<palette::Lab> {
        let mut octree = OctreePaletteBuilder::<PALETTE_SIZE>::new();

        for pixel in image.pixels() {
            octree.insert(Srgb::<u8>::new(pixel[0], pixel[1], pixel[2]));
        }

        let mut candidate_merges = octree
            .nodes
            .par_iter()
            .enumerate()
            .rev()
            .filter_map(|(idx, node)| {
                (node.depth == 7).then_some(Candidate::<USE_MIN_HEAP> {
                    index: idx,
                    count: node.count,
                })
            })
            .collect::<BinaryHeap<_>>();

        while candidate_merges.len() > Self::PALETTE_SIZE {
            let Some(min_candidate) = candidate_merges.pop() else {
                break;
            };

            let Node {
                rgb,
                count,
                parent: parent_idx,
                ..
            } = octree.nodes[min_candidate.index];

            let parent = &mut octree.nodes[parent_idx];
            parent.rgb.0 += rgb.0;
            parent.rgb.1 += rgb.1;
            parent.rgb.2 += rgb.2;
            parent.count += count;
            let child_idx = parent
                .children
                .iter_mut()
                .position(|child| child.is_some() && child.unwrap() == min_candidate.index);
            parent.children[child_idx.unwrap()] = None;

            if parent.children.iter().all(|child| child.is_none()) {
                candidate_merges.push(Candidate {
                    index: parent_idx,
                    count: parent.count,
                });
            }
        }

        candidate_merges
            .into_par_iter()
            .map(|node| {
                let node = &octree.nodes[node.index];
                let lab: Lab = Srgb::new(
                    (node.rgb.0 / node.count as u64) as u8,
                    (node.rgb.1 / node.count as u64) as u8,
                    (node.rgb.2 / node.count as u64) as u8,
                )
                .into_format()
                .into_color();

                [
                    OrderedFloat(lab.l),
                    OrderedFloat(lab.a),
                    OrderedFloat(lab.b),
                ]
            })
            .collect::<HashSet<_>>()
            .into_iter()
            .map(|[l, a, b]| Lab::new(*l, *a, *b))
            .collect()
    }
}

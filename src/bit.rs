//! Encodes a palette by bucketing bit ranges into a power of two number of
//! buckets. This is very fast and produces ok results for most images at larger
//! palette sizes (e.g. 256).

use std::cell::RefCell;

use dilate::DilateExpand;
use image::Rgba;
use palette::IntoColor;
use palette::Lab;
use palette::Srgb;
use rayon::iter::ParallelIterator;

/// Builds a palette by bucketing pixel colors via bit-dilation into a
/// power-of-two number of buckets and averaging the colors in each bucket.
///
/// This is the fastest palette builder and produces acceptable results at
/// larger palette sizes (e.g. 256).
pub struct BitPaletteBuilder {
    pub(crate) shift: usize,
}

impl BitPaletteBuilder {
    pub(crate) fn new(palette_size: usize) -> Self {
        BitPaletteBuilder {
            shift: Self::shift(palette_size),
        }
    }

    pub(crate) fn shift(palette_size: usize) -> usize {
        24 - palette_size.ilog2() as usize
    }

    pub(crate) fn index(
        color: Srgb<u8>,
        shift: usize,
    ) -> usize {
        let r = color.red.dilate_expand::<3>().value();
        let g = color.green.dilate_expand::<3>().value();
        let b = color.blue.dilate_expand::<3>().value();

        // Since elements to the right will get shifted off first, we put them in grb
        // order (order of most-least significant for luminance). This probably doesn't
        // make a huge difference, but the theory is nice.
        let rgb = g << 2 | r << 1 | b;

        (rgb >> shift) as usize
    }

    /// Quantize the image into `palette_size` colors using bit-dilation
    /// bucketing and return the resulting palette in Lab color space.
    pub fn build_palette(
        image: &image::RgbaImage,
        palette_size: usize,
    ) -> Vec<Lab> {
        let builder = Self::new(palette_size);

        thread_local! {
            static PALETTE: RefCell<Vec<(u64, u64, u64, u64)>> = RefCell::default();
        }

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(rayon::current_num_threads())
            .build()
            .unwrap();

        pool.install(|| {
            image.par_pixels().for_each(|pixel| {
                PALETTE.with_borrow_mut(|palette| {
                    palette.resize(palette_size, (0, 0, 0, 0));

                    let pixel = Srgb::<u8>::new(pixel[0], pixel[1], pixel[2]);
                    let index = Self::index(pixel, builder.shift);
                    palette[index].0 += pixel.red as u64;
                    palette[index].1 += pixel.green as u64;
                    palette[index].2 += pixel.blue as u64;
                    palette[index].3 += 1;
                });
            });
        });

        let per_thread_palettes = pool.broadcast(|_ctx| PALETTE.with_borrow_mut(std::mem::take));

        let mut final_palette = vec![(0, 0, 0, 0); palette_size];
        for palette in per_thread_palettes {
            for (dest, src) in final_palette.iter_mut().zip(palette) {
                dest.0 += src.0;
                dest.1 += src.1;
                dest.2 += src.2;
                dest.3 += src.3;
            }
        }

        final_palette
            .into_iter()
            .filter(|node| node.3 > 0)
            .map(|node| {
                let rgb = Srgb::new(
                    (node.0 / node.3) as u8,
                    (node.1 / node.3) as u8,
                    (node.2 / node.3) as u8,
                );
                rgb.into_format().into_color()
            })
            .collect::<Vec<_>>()
    }

    pub(crate) fn build_bucketer(
        palette: &[Lab],
        palette_size: usize,
    ) -> BitPaletteBucketer {
        BitPaletteBucketer::new(palette, palette_size)
    }
}

/// A [`PaletteBucketer`](crate::dither::PaletteBucketer) variant that maps
/// pixels directly to palette entries via the same bit-dilation bucketing used
/// by [`BitPaletteBuilder`]. This avoids both the Lab color-space conversion
/// and the KD-tree query for each pixel, making it significantly faster than
/// [`KdTreeBucketer`](crate::dither::KdTreeBucketer) when used with
/// [`Dither::None`](crate::dither::Dither::None).
pub struct BitPaletteBucketer {
    lut: Vec<(usize, usize, f32, f32)>,
    shift: usize,
}

impl BitPaletteBucketer {
    /// Build a lookup table that maps each bit-bucket index to its two nearest
    /// palette entries with distances. Each palette Lab entry is converted to
    /// Srgb to determine which bucket it belongs to, giving us a representative
    /// color per bucket without needing to re-scan the image.
    fn new(
        palette: &[Lab],
        palette_size: usize,
    ) -> Self {
        let shift = BitPaletteBuilder::shift(palette_size);
        let n_bits = palette_size.ilog2() as usize;
        let masks = morton_dim_masks(n_bits);
        let all_mask = palette_size - 1;

        let mut bucket_entries: Vec<Vec<usize>> = vec![vec![]; palette_size];
        for (pi, &lab) in palette.iter().enumerate() {
            let rgb: Srgb = lab.into_color();
            let rgb: Srgb<u8> = rgb.into_format();
            let idx = BitPaletteBuilder::index(rgb, shift);
            bucket_entries[idx].push(pi);
        }

        let lut = (0..palette_size)
            .map(|bucket_idx| {
                if bucket_entries[bucket_idx].is_empty() {
                    return (0, 0, 0.0, 0.0);
                }

                let nearest = bucket_entries[bucket_idx][0];
                let second = find_second_nearest(
                    nearest,
                    bucket_idx,
                    &bucket_entries,
                    palette,
                    masks,
                    all_mask,
                );
                (nearest, second.0, 0.0, second.1)
            })
            .collect();

        Self { lut, shift }
    }

    pub(crate) fn nearest(
        &self,
        point: &[f32; 3],
    ) -> usize {
        let lab = Lab::new(point[0], point[1], point[2]);
        let rgb: Srgb = lab.into_color();
        let rgb: Srgb<u8> = rgb.into_format();
        self.lut[BitPaletteBuilder::index(rgb, self.shift)].0
    }

    pub(crate) fn nearest_two(
        &self,
        point: Rgba<u8>,
    ) -> [(usize, f32); 2] {
        let (n1, n2, d1, d2) =
            self.lut[BitPaletteBuilder::index(Srgb::new(point[0], point[1], point[2]), self.shift)];
        [(n1, d1), (n2, d2)]
    }

    pub(crate) fn nearest_rgb(
        &self,
        pixel: Rgba<u8>,
    ) -> usize {
        let index = BitPaletteBuilder::index(Srgb::new(pixel[0], pixel[1], pixel[2]), self.shift);
        self.lut[index].0
    }
}

/// Bit masks for each dimension (B, R, G) in an `n`-bit Morton code
/// with the interleaving order `...g r b g r b`.
fn morton_dim_masks(n: usize) -> [usize; 3] {
    let mut masks = [0usize; 3]; // [B, R, G]
    for i in 0..n {
        masks[i % 3] |= 1 << i;
    }
    masks
}

/// Increment one dimension of a Morton code. Returns `None` on overflow.
fn morton_inc(
    z: usize,
    dim_mask: usize,
    all_mask: usize,
) -> Option<usize> {
    let not_dim = all_mask & !dim_mask;
    let t = (z | not_dim) + 1;
    if t > all_mask {
        return None;
    }
    Some((t & dim_mask) | (z & not_dim))
}

/// Decrement one dimension of a Morton code. Returns `None` on underflow.
fn morton_dec(
    z: usize,
    dim_mask: usize,
    all_mask: usize,
) -> Option<usize> {
    if z & dim_mask == 0 {
        return None;
    }
    let not_dim = all_mask & !dim_mask;
    let t = (z & dim_mask).wrapping_sub(1);
    Some((t & dim_mask) | (z & not_dim))
}

/// Apply ±1 offsets in each Morton dimension. Returns `None` if any
/// dimension would go out of bounds.
fn morton_neighbor(
    z: usize,
    deltas: [i32; 3],
    masks: &[usize; 3],
    all_mask: usize,
) -> Option<usize> {
    let mut result = z;
    for (delta, &mask) in deltas.iter().zip(masks.iter()) {
        match delta.signum() {
            1 => result = morton_inc(result, mask, all_mask)?,
            -1 => result = morton_dec(result, mask, all_mask)?,
            _ => {}
        }
    }
    Some(result)
}

/// Find the second-nearest palette entry to `palette[nearest]` by searching
/// the 26 neighboring Morton cells. Falls back to a full palette scan when
/// no neighbor contains a candidate.
fn find_second_nearest(
    nearest: usize,
    bucket_idx: usize,
    bucket_entries: &[Vec<usize>],
    palette: &[Lab],
    masks: [usize; 3],
    all_mask: usize,
) -> (usize, f32) {
    use palette::color_difference::EuclideanDistance;

    let target = &palette[nearest];
    let mut best = (0usize, f32::INFINITY);

    for db in -1i32..=1 {
        for dr in -1i32..=1 {
            for dg in -1i32..=1 {
                let nb = if db == 0 && dr == 0 && dg == 0 {
                    Some(bucket_idx)
                } else {
                    morton_neighbor(bucket_idx, [db, dr, dg], &masks, all_mask)
                };
                if let Some(nb) = nb {
                    for &pi in &bucket_entries[nb] {
                        if pi == nearest {
                            continue;
                        }
                        let dist = target.distance_squared(palette[pi]);
                        if dist < best.1 {
                            best = (pi, dist);
                        }
                    }
                }
            }
        }
    }

    if best.1.is_finite() {
        return best;
    }

    for (pi, color) in palette.iter().enumerate() {
        if pi == nearest {
            continue;
        }
        let dist = target.distance_squared(*color);
        if dist < best.1 {
            best = (pi, dist);
        }
    }

    best
}

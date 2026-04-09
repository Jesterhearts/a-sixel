use a_sixel::SixelEncoder;

const IMAGE: &[u8] = include_bytes!("transparent.png");

fn main() {
    let image = image::load_from_memory(IMAGE).unwrap();
    let six = SixelEncoder::default().encode(image.to_rgba8());
    println!("{six}");
}

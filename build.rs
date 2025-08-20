use rustc_version::{
    Channel,
    version_meta,
};

fn main() {
    println!("cargo::rustc-check-cfg=cfg(ENABLE_DOC_AUTO_CFG)");
    let enable_doc_auto_cfg = match version_meta().unwrap().channel {
        Channel::Nightly => "ENABLE_DOC_AUTO_CFG",
        _ => "DISABLE_DOC_AUTO_CFG",
    };
    println!("cargo:rustc-cfg={}", enable_doc_auto_cfg);
}

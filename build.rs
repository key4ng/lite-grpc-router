fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(false) // client only
        .compile_protos(
            &["proto/sglang_scheduler.proto", "proto/common.proto"],
            &["proto/"],
        )?;
    Ok(())
}

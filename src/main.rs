mod client;

// Re-export smg module at crate root so generated sglang proto code
// can resolve cross-package references via super::super::super::super::smg::...
pub mod smg {
    pub mod grpc {
        pub mod common {
            tonic::include_proto!("smg.grpc.common");
        }
    }
}

fn main() {
    println!("lite-grpc");
}

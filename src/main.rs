mod client;
mod parsers;
mod tokenizer;
mod types;
mod worker;

// Proto modules: hierarchy must match proto package names so tonic's
// generated cross-package super:: references resolve correctly.
pub mod smg {
    pub mod grpc {
        pub mod common {
            tonic::include_proto!("smg.grpc.common");
        }
    }
}
pub mod sglang {
    pub mod grpc {
        pub mod scheduler {
            tonic::include_proto!("sglang.grpc.scheduler");
        }
    }
}

// Convenience re-exports
pub mod proto {
    pub use crate::sglang::grpc::scheduler as sglang;
    pub use crate::smg::grpc::common;
}

fn main() {
    println!("lite-grpc");
}

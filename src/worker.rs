use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::{bail, Result};

use crate::client::SglangClient;

pub struct Worker {
    pub client: SglangClient,
    pub model_name: String,
    healthy: AtomicBool,
}

impl Worker {
    pub fn new(client: SglangClient, model_name: String) -> Self {
        Self {
            client,
            model_name,
            healthy: AtomicBool::new(true),
        }
    }

    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Relaxed)
    }

    pub fn mark_unhealthy(&self) {
        self.healthy.store(false, Ordering::Relaxed);
    }

    pub fn mark_healthy(&self) {
        self.healthy.store(true, Ordering::Relaxed);
    }
}

pub struct WorkerPool {
    workers: Vec<Arc<Worker>>,
    next: AtomicUsize,
}

impl WorkerPool {
    pub fn new(workers: Vec<Arc<Worker>>) -> Self {
        Self {
            workers,
            next: AtomicUsize::new(0),
        }
    }

    /// Round-robin select a healthy worker.
    pub async fn select(&self) -> Result<Arc<Worker>> {
        let len = self.workers.len();

        let start = self.next.fetch_add(1, Ordering::Relaxed);
        for i in 0..len {
            let idx = (start + i) % len;
            if self.workers[idx].is_healthy() {
                return Ok(Arc::clone(&self.workers[idx]));
            }
        }

        // All unhealthy — attempt recovery
        for worker in &self.workers {
            if let Ok(true) = worker.client.health_check().await {
                worker.mark_healthy();
                return Ok(Arc::clone(worker));
            }
        }

        bail!("no healthy workers available")
    }

    pub fn workers(&self) -> &[Arc<Worker>] {
        &self.workers
    }

    pub fn model_name(&self) -> &str {
        &self.workers[0].model_name
    }
}

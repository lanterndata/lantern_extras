use crate::types::AnyhowVoidResult;
use sysinfo::{System, SystemExt};

pub fn bytes_to_integer_le<T>(bytes: &[u8]) -> T
where
    T: From<u8> + std::ops::Shl<usize, Output = T> + std::ops::BitOr<Output = T> + Default,
{
    let mut result: T = Default::default();

    for &byte in bytes.iter().rev() {
        result = (result << 8) | T::from(byte);
    }

    result
}

fn level_probabilities(m: f64) -> Vec<f64> {
    let m_l = 1.0 / m.ln();
    let mut res: Vec<f64> = Vec::new();
    for level in 0.. {
        let p = (-level as f64 / m_l).exp() * (1.0 - (-1.0 / m_l).exp());
        if p < 1e-12 {
            break;
        }
        res.push(p);
    }

    let sum = res.iter().sum::<f64>();
    let response_len = res.len();
    res[response_len - 1] += 1.0 - sum;
    res
}

fn estimate_memory(
    num_vectors: usize,
    hnsw_m: usize,
    vector_dim: usize,
    pq: bool,
    pq_num_clusters: usize,
    pq_num_subvectors: usize,
) -> f64 {
    const VECTOR_SCALAR_SIZE: usize = 32;
    const BYTES_PER_NEIGHBOR_ID: usize = 6;
    const BITS_PER_BYTE: usize = 8;

    let bits_per_neighbor_id: usize = BYTES_PER_NEIGHBOR_ID * BITS_PER_BYTE;

    let vector_data_size = vector_dim * VECTOR_SCALAR_SIZE / BITS_PER_BYTE * num_vectors;

    let per_cluster_id_bytes = (pq_num_clusters + 255) / 256;
    let pq_vector_data_size = pq_num_subvectors * per_cluster_id_bytes * num_vectors;

    let level_ps = level_probabilities(hnsw_m as f64);
    let total_size: Vec<f64> = level_ps
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let node_size_bytes = ((i + 2) * hnsw_m * bits_per_neighbor_id) / BITS_PER_BYTE;
            let num_nodes = (num_vectors as f64 * v).floor() as usize;
            node_size_bytes as f64 * num_nodes as f64
        })
        .collect();

    let neighbor_metadata_size: f64 = total_size.iter().sum();

    (pq as usize * pq_vector_data_size as usize + !pq as usize * vector_data_size) as f64
        + neighbor_metadata_size
}

pub fn check_available_memory_for_index(
    num_vectors: usize,
    hnsw_m: usize,
    vector_dim: usize,
    pq: bool,
    pq_num_clusters: usize,
    pq_num_subvectors: usize,
) -> AnyhowVoidResult {
    let estimated_memory = estimate_memory(
        num_vectors,
        hnsw_m,
        vector_dim,
        pq,
        pq_num_clusters,
        pq_num_subvectors,
    );

    let mut sys = System::new_all();
    sys.refresh_all();
    let total_free_mem =
        (sys.total_memory() - sys.used_memory()) + (sys.total_swap() - sys.used_swap());
    let total_free_mem = total_free_mem as f64;

    println!("Free mem {total_free_mem}, Estimated memory {estimated_memory}");
    if total_free_mem < estimated_memory {
        let mem_needed = estimated_memory as usize / 1024 / 1024 / 1024;
        let mem_avail = total_free_mem as usize / 1024 / 1024 / 1024;
        anyhow::bail!("Not enough free memory to construct HNSW index. Memory required {mem_needed}GB, memory available {mem_avail}GB")
    }

    Ok(())
}

use crate::{
    types::AnyhowVoidResult,
    utils::{get_full_table_name, quote_ident},
};
use postgres::Transaction;
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

    if total_free_mem < estimated_memory {
        let mem_needed = estimated_memory as usize / 1024 / 1024 / 1024;
        let mem_avail = total_free_mem as usize / 1024 / 1024 / 1024;
        anyhow::bail!("Not enough free memory to construct HNSW index. Memory required {mem_needed}GB, memory available {mem_avail}GB")
    }

    Ok(())
}

pub trait UnifiedClient<'a> {
    fn infer_column_dimensions(
        &mut self,
        full_table_name: &str,
        column: &str,
    ) -> Result<Option<usize>, anyhow::Error>;
    fn codebook_exists(&mut self, table_name: &str) -> Result<bool, anyhow::Error>;
    fn get_centroid_count(&mut self, full_table_name: &str) -> Result<usize, anyhow::Error>;
    fn get_subvector_count(&mut self, full_table_name: &str) -> Result<usize, anyhow::Error>;
    fn get_codebook_data(
        &mut self,
        full_table_name: &str,
    ) -> Result<Vec<(i32, i32, Vec<f32>)>, anyhow::Error>;
}

pub fn get_infer_dims_query(full_table_name: &str, column: &str) -> String {
    format!("SELECT ARRAY_LENGTH({col}, 1) as dim FROM {full_table_name} WHERE {col} IS NOT NULL LIMIT 1",col=quote_ident(column))
}

pub fn get_codebook_exists_query(table_name: &str) -> String {
    format!("SELECT true FROM information_schema.tables WHERE table_schema='_lantern_internal' AND table_name='{table_name}'")
}

pub fn get_centroid_count_query(full_table_name: &str) -> String {
    format!("SELECT COUNT(*) as cnt FROM {full_table_name} WHERE subvector_id = 0")
}

pub fn get_subvector_count_query(full_table_name: &str) -> String {
    format!("SELECT COUNT(*) as cnt FROM {full_table_name} WHERE centroid_id = 0")
}

pub fn get_codebook_data_query(full_table_name: &str) -> String {
    format!("SELECT subvector_id, centroid_id, c FROM {full_table_name}")
}

pub fn get_count_estimation_query(full_table_name: &str, column: &str) -> String {
    format!(
        "SELECT COUNT(*) as cnt FROM {full_table_name} WHERE {} IS NOT NULL",
        quote_ident(column)
    )
}

pub fn get_portal_query(full_table_name: &str, column: &str) -> String {
    format!(
        "SELECT ctid, {col} as v FROM {full_table_name} WHERE {col} IS NOT NULL",
        col = quote_ident(column)
    )
}

impl<'a> UnifiedClient<'a> for Transaction<'a> {
    fn infer_column_dimensions(
        &mut self,
        full_table_name: &str,
        column: &str,
    ) -> Result<Option<usize>, anyhow::Error> {
        let rows = self.query(&get_infer_dims_query(full_table_name, column), &[])?;

        if rows.len() == 0 {
            return Ok(None);
        }

        Ok(Some(rows.first().unwrap().get::<usize, i32>(0) as usize))
    }

    fn codebook_exists(&mut self, table_name: &str) -> Result<bool, anyhow::Error> {
        let rows_codebook_exists = self.query(&get_codebook_exists_query(table_name), &[])?;

        Ok(rows_codebook_exists.len() > 0)
    }

    fn get_centroid_count(&mut self, full_table_name: &str) -> Result<usize, anyhow::Error> {
        let rows = self.query(&get_centroid_count_query(full_table_name), &[])?;
        Ok(rows.first().unwrap().get::<usize, i64>(0) as usize)
    }

    fn get_subvector_count(&mut self, full_table_name: &str) -> Result<usize, anyhow::Error> {
        let rows = self.query(&get_subvector_count_query(full_table_name), &[])?;
        Ok(rows.first().unwrap().get::<usize, i64>(0) as usize)
    }

    fn get_codebook_data(
        &mut self,
        full_table_name: &str,
    ) -> Result<Vec<(i32, i32, Vec<f32>)>, anyhow::Error> {
        let rows = self.query(&get_codebook_data_query(full_table_name), &[])?;

        Ok(rows
            .iter()
            .map(|r| (r.get(0), r.get(1), r.get(2)))
            .collect())
    }
}

pub fn get_and_validate_dimensions<'a>(
    full_table_name: &str,
    column: &str,
    dimensions: usize,
    client: &mut impl UnifiedClient<'a>,
) -> Result<usize, anyhow::Error> {
    let infered_dimensions = client.infer_column_dimensions(full_table_name, column)?;

    if infered_dimensions.is_none() {
        anyhow::bail!("Cannot create an external index on empty table");
    }

    let infered_dimensions = infered_dimensions.unwrap();

    if dimensions != 0 && infered_dimensions != dimensions {
        // I didn't complitely remove the dimensions from args
        // To have extra validation when reindexing external index
        // This is invariant and should never be a case
        anyhow::bail!("Infered dimensions ({infered_dimensions}) does not match with the provided dimensions ({dimensions})");
    }

    if infered_dimensions == 0 {
        anyhow::bail!("Column does not have dimensions");
    }

    Ok(infered_dimensions)
}

pub fn get_codebook<'a>(
    table: &str,
    column: &str,
    dimensions: usize,
    client: &mut impl UnifiedClient<'a>,
) -> Result<(Vec<f32>, usize, usize), anyhow::Error> {
    let mut v: Vec<f32> = vec![];
    let codebook_table_name = format!("pq_{table}_{column}",);
    let full_codebook_table_name = get_full_table_name("_lantern_internal", &codebook_table_name);

    let codebook_exists = client.codebook_exists(&codebook_table_name)?;
    if !codebook_exists {
        anyhow::bail!("Codebook table {full_codebook_table_name} does not exist");
    }

    let num_centroids = client.get_centroid_count(&full_codebook_table_name)?;
    let num_subvectors = client.get_subvector_count(&full_codebook_table_name)?;

    if num_centroids == 0 || num_subvectors == 0 {
        anyhow::bail!("Invalid codebook table");
    }

    v.resize(num_centroids * dimensions, 0.);

    let rows = client.get_codebook_data(&full_codebook_table_name)?;

    for r in rows {
        let subvector_id: i32 = r.0;
        let centroid_id: i32 = r.1;
        let subvector: Vec<f32> = r.2;
        for i in 0..subvector.len() {
            v[centroid_id as usize * dimensions + subvector_id as usize * subvector.len() + i] =
                subvector[i];
        }
    }

    Ok((v, num_centroids, num_subvectors))
}

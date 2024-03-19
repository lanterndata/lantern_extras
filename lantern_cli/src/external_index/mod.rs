use rand::Rng;
use std::io::BufWriter;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, SyncSender};
use std::sync::{mpsc, RwLock};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;
use std::{fs, io};
pub use usearch::ffi::{IndexOptions, ScalarKind};
use usearch::Index;

use crate::logger::{LogLevel, Logger};
use crate::types::*;
use crate::utils::{get_full_table_name, quote_ident};
use postgres::{Client, NoTls};
use postgres_large_objects::LargeObject;
use postgres_types::FromSql;

use self::utils::{
    check_available_memory_for_index, get_and_validate_dimensions, get_codebook,
    get_count_estimation_query, get_portal_query,
};

pub mod cli;
mod postgres_large_objects;
pub mod utils;

// Used to control chunk size when copying index file to postgres server
static COPY_BUFFER_CHUNK_SIZE: usize = 1024 * 1024 * 10; // 10MB

type IndexItem = (u64, Vec<f32>);

#[derive(Debug)]
struct Tid {
    label: u64,
}

impl<'a> FromSql<'a> for Tid {
    fn from_sql(
        _: &postgres_types::Type,
        raw: &'a [u8],
    ) -> Result<Self, Box<dyn std::error::Error + Sync + Send>> {
        let mut bytes: Vec<u8> = Vec::with_capacity(raw.len());

        // Copy bytes of block_number->bi_hi first 2 bytes
        for b in raw[..2].iter().rev() {
            bytes.push(*b);
        }

        // Copy bytes of block_number->bi_lo next 2 bytes
        for b in raw[2..4].iter().rev() {
            bytes.push(*b);
        }

        // Copy bytes of index_number last 2 bytes
        for b in raw[4..6].iter().rev() {
            bytes.push(*b);
        }

        let label: u64 = utils::bytes_to_integer_le(&bytes);
        Ok(Tid { label })
    }

    fn accepts(ty: &postgres_types::Type) -> bool {
        ty.name() == "tid"
    }
}

pub struct ThreadSafeIndex {
    inner: Index,
}

impl ThreadSafeIndex {
    fn add(&self, label: u64, data: &[f32]) -> AnyhowVoidResult {
        self.inner.add(label, data)?;
        Ok(())
    }
    pub fn save(&self, path: &str) -> AnyhowVoidResult {
        self.inner.save(path)?;
        Ok(())
    }
}

unsafe impl Sync for ThreadSafeIndex {}
unsafe impl Send for ThreadSafeIndex {}

fn index_chunk(rows: Vec<IndexItem>, index: Arc<ThreadSafeIndex>) -> AnyhowVoidResult {
    for row in rows {
        index.add(row.0, &row.1)?;
    }
    Ok(())
}

fn report_progress(progress_cb: &Option<ProgressCbFn>, logger: &Logger, progress: u8) {
    logger.info(&format!("Progress {progress}%"));
    if progress_cb.is_some() {
        let cb = progress_cb.as_ref().unwrap();
        cb(progress);
    }
}

pub fn create_index_from_stream(
    logger: Arc<Logger>,
    is_canceled: Option<Arc<RwLock<bool>>>,
    progress_cb: Option<ProgressCbFn>,
    should_create_index: bool,
    options: &IndexOptions,
    row_count: usize,
) -> Result<
    (
        SyncSender<Vec<IndexItem>>,
        Sender<u8>,
        Arc<ThreadSafeIndex>,
        Vec<JoinHandle<AnyhowVoidResult>>,
    ),
    anyhow::Error,
> {
    check_available_memory_for_index(
        row_count,
        options.connectivity,
        options.dimensions,
        options.pq_construction,
        options.num_centroids,
        options.num_threads,
    )?;

    let mut handles = vec![];
    let num_cores: usize = std::thread::available_parallelism().unwrap().into();
    logger.info(&format!("Number of available CPU cores: {}", num_cores));
    let is_canceled = is_canceled.unwrap_or(Arc::new(RwLock::new(false)));
    let (progress_tx, progress_rx): (Sender<u8>, Receiver<u8>) = mpsc::channel();
    let (tx, rx): (SyncSender<Vec<IndexItem>>, Receiver<Vec<IndexItem>>) =
        mpsc::sync_channel(num_cores);
    let rx_arc = Arc::new(Mutex::new(rx));
    let progress_logger = logger.clone();
    let index = Index::new(options)?;
    // reserve enough memory on index
    index.reserve(row_count)?;
    let thread_safe_index = ThreadSafeIndex { inner: index };

    logger.info(&format!("Items to index {}", row_count));

    let index_arc = Arc::new(thread_safe_index);

    std::thread::spawn(move || -> AnyhowVoidResult {
        let mut prev_progress = 0;
        for progress in progress_rx {
            if progress == prev_progress {
                continue;
            }
            prev_progress = progress;
            report_progress(&progress_cb, &progress_logger, progress);

            if progress == 100 {
                break;
            }
        }
        Ok(())
    });

    let processed_cnt = Arc::new(AtomicU64::new(0));
    for _ in 0..num_cores {
        // spawn thread
        let index_ref = index_arc.clone();
        let receiver = rx_arc.clone();
        let is_canceled = is_canceled.clone();
        let progress_tx = progress_tx.clone();
        let processed_cnt = processed_cnt.clone();

        let handle = std::thread::spawn(move || -> AnyhowVoidResult {
            loop {
                let lock = receiver.lock();

                if let Err(e) = lock {
                    anyhow::bail!("{e}");
                }

                let rx = lock.unwrap();
                let rows = rx.recv();
                // release the lock so other threads can take rows
                drop(rx);

                if rows.is_err() {
                    // channel has been closed
                    break;
                }

                if *is_canceled.read().unwrap() {
                    // This variable will be changed from outside to gracefully
                    // exit job on next chunk
                    anyhow::bail!(JOB_CANCELLED_MESSAGE);
                }

                let rows = rows.unwrap();
                let rows_cnt = rows.len();
                index_chunk(rows, index_ref.clone())?;
                let all_count = processed_cnt.fetch_add(rows_cnt as u64, Ordering::SeqCst);
                let mut progress = (all_count as f64 / row_count as f64 * 100.0) as u8;
                if should_create_index {
                    // reserve 20% progress for index import
                    progress = if progress > 20 { progress - 20 } else { 0 };
                }

                if progress > 0 {
                    progress_tx.send(progress)?;
                }
            }
            Ok(())
        });
        handles.push(handle);
    }

    Ok((tx, progress_tx, index_arc, handles))
}

pub fn create_usearch_index(
    args: &cli::CreateIndexArgs,
    progress_cb: Option<ProgressCbFn>,
    is_canceled: Option<Arc<RwLock<bool>>>,
    logger: Option<Logger>,
) -> Result<(), anyhow::Error> {
    let logger = Arc::new(logger.unwrap_or(Logger::new("Lantern Index", LogLevel::Debug)));
    let total_start_time = Instant::now();

    // get all row count
    let mut client = Client::connect(&args.uri, NoTls)?;
    let mut transaction = client.transaction()?;
    let full_table_name = get_full_table_name(&args.schema, &args.table);

    transaction.execute("SET lock_timeout='5s'", &[])?;
    transaction.execute(
        &format!("LOCK TABLE ONLY {full_table_name} IN SHARE MODE"),
        &[],
    )?;

    let dimensions =
        get_and_validate_dimensions(&full_table_name, &args.column, args.dims, &mut transaction)?;

    logger.info(&format!(
        "Creating index with parameters dimensions={} m={} ef={} ef_construction={}",
        dimensions, args.m, args.ef, args.efc
    ));

    let mut pq_codebook: *const f32 = std::ptr::null();
    let v: Vec<f32>;
    let mut num_centroids: usize = 0;
    let mut num_subvectors: usize = 0;

    if args.pq {
        let (codebook_vector, count_c, count_sv) =
            get_codebook(&args.table, &args.column, dimensions, &mut transaction)?;
        v = codebook_vector;
        num_centroids = count_c;
        num_subvectors = count_sv;

        logger.info(&format!(
            "Codebook has {} rows - {num_centroids} centroids and {num_subvectors} subvectors",
            v.len()
        ));

        pq_codebook = v.as_ptr();
    }

    let options = IndexOptions {
        dimensions,
        metric: args.metric_kind.value(),
        quantization: ScalarKind::F32,
        multi: false,
        connectivity: args.m,
        expansion_add: args.efc,
        expansion_search: args.ef,

        num_threads: 0, // automatic

        // note: pq_construction and pq_output distinction is not yet implemented in usearch
        // in the future, if pq_construction is false, we will use full vectors in memory (and
        // require large memory for construction) but will output pq-quantized graph
        //
        // currently, regardless of pq_construction value, as long as pq_output is true,
        // we construct a pq_quantized index using quantized values during construction
        pq_construction: args.pq,
        pq_output: args.pq,
        num_centroids,
        num_subvectors,
        codebook: pq_codebook,
    };

    let should_create_index = args.import;
    let start_time = Instant::now();

    let rows = transaction.query(
        &get_count_estimation_query(&full_table_name, &args.column),
        &[],
    )?;

    logger.debug(&format!(
        "Count estimation took {}s",
        start_time.elapsed().as_secs()
    ));

    let start_time = Instant::now();
    let count: i64 = rows[0].get(0);

    let (tx, progress_tx, index_arc, handles) = create_index_from_stream(
        logger.clone(),
        is_canceled.clone(),
        progress_cb,
        should_create_index,
        &options,
        count as usize,
    )?;

    let is_canceled = is_canceled.unwrap_or(Arc::new(RwLock::new(false)));
    // With portal we can execute a query and poll values from it in chunks
    let portal = transaction.bind(&get_portal_query(&full_table_name, &args.column), &[])?;

    loop {
        // poll 2000 rows from portal and send it to worker threads via channel
        let rows = transaction.query_portal(&portal, 2000)?;
        if rows.len() == 0 {
            break;
        }
        if *is_canceled.read().unwrap() {
            // This variable will be changed from outside to gracefully
            // exit job on next chunk
            anyhow::bail!(JOB_CANCELLED_MESSAGE);
        }
        tx.send(
            rows.iter()
                .map(|r| (r.get::<usize, Tid>(0).label, r.get::<usize, Vec<f32>>(1)))
                .collect(),
        )?;
    }

    // Exit all channels
    drop(tx);

    // Wait for all threads to finish processing
    for handle in handles {
        if let Err(e) = handle.join() {
            logger.error("{e}");
            anyhow::bail!("{:?}", e);
        }
    }

    logger.debug(&format!(
        "Indexing took {}s",
        start_time.elapsed().as_secs()
    ));

    index_arc.save(&args.out)?;
    logger.info(&format!(
        "Index saved under {} in {}s",
        &args.out,
        start_time.elapsed().as_secs()
    ));

    drop(index_arc);
    drop(portal);

    if args.import {
        let op_class = args.metric_kind.to_ops();
        if args.remote_database {
            let start_time = Instant::now();
            logger.info("Copying index file into database server...");
            let mut rng = rand::thread_rng();
            let data_dir = transaction.query_one("SHOW data_directory", &[])?;
            let data_dir: String = data_dir.try_get(0)?;
            let index_path = format!("{data_dir}/ldb-index-{}.usearch", rng.gen_range(0..1000));
            let mut large_object = LargeObject::new(transaction, &index_path);
            large_object.create()?;
            let mut reader = fs::File::open(Path::new(&args.out))?;
            let mut buf_writer =
                BufWriter::with_capacity(COPY_BUFFER_CHUNK_SIZE, &mut large_object);
            io::copy(&mut reader, &mut buf_writer)?;
            logger.debug(&format!(
                "Index copy to database took {}s",
                start_time.elapsed().as_secs()
            ));
            progress_tx.send(90)?;
            drop(reader);
            drop(buf_writer);
            logger.info("Creating index from file...");
            let start_time = Instant::now();
            large_object.finish(
                &get_full_table_name(&args.schema, &args.table),
                &quote_ident(&args.column),
                args.index_name.as_deref(),
                &op_class,
                args.ef,
                args.efc,
                dimensions,
                args.m,
                args.pq,
            )?;
            logger.debug(&format!(
                "Index import took {}s",
                start_time.elapsed().as_secs()
            ));
            fs::remove_file(Path::new(&args.out))?;
        } else {
            // If job is run on the same server as database we can skip copying part
            progress_tx.send(90)?;
            logger.info("Creating index from file...");
            let start_time = Instant::now();

            let mut idx_name = "".to_owned();

            if let Some(name) = &args.index_name {
                idx_name = quote_ident(name);
                transaction.execute(&format!("DROP INDEX IF EXISTS {idx_name}"), &[])?;
            }

            transaction.execute(
            &format!("CREATE INDEX {idx_name} ON {table_name} USING lantern_hnsw({column_name} {op_class}) WITH (_experimental_index_path='{index_path}', pq={pq}, ef={ef}, dim={dim}, m={m}, ef_construction={ef_construction});", index_path=args.out, table_name=&get_full_table_name(&args.schema, &args.table),
            column_name=&quote_ident(&args.column), pq=args.pq, m=args.m, ef=args.ef, ef_construction=args.efc, dim=dimensions),
            &[],
            )?;

            transaction.commit()?;
            logger.debug(&format!(
                "Index import took {}s",
                start_time.elapsed().as_secs()
            ));
            fs::remove_file(Path::new(&args.out))?;
        }
        progress_tx.send(100)?;
        logger.info(&format!(
            "Index imported to table {} and removed from filesystem",
            &args.table
        ));
        logger.debug(&format!(
            "Total indexing took {}s",
            total_start_time.elapsed().as_secs()
        ));
    }

    Ok(())
}

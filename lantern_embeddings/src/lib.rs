use csv::Writer;
use lantern_embeddings_core::clip;
use lantern_logger::{LogLevel, Logger};
use rand::Rng;
use std::io::Write;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{mpsc, Arc};
use std::thread::JoinHandle;
use std::time::Instant;
use std::{panic, process};

use postgres::{Client, NoTls, Row};

pub mod cli;

type EmbeddingRecord = (String, Vec<f32>);
type AnyhowVoidResult = Result<(), anyhow::Error>;

// This function will do the following
// 1. Get approximate number of rows from pg_class (this is just for info logging)
// 2. Create transaction portal which will poll data from database of batch size provided via args
// 3. Send the rows over the channel
fn producer_worker(
    args: &cli::EmbeddingArgs,
    tx: Sender<Vec<Row>>,
    logger: Arc<Logger>,
) -> Result<JoinHandle<AnyhowVoidResult>, anyhow::Error> {
    let uri = args.uri.clone();
    let pk = args.pk.clone();
    let column = args.column.clone();
    let schema = args.schema.clone();
    let table = args.table.clone();
    let batch_size = args.batch_size.clone();

    let handle = std::thread::spawn(move || {
        let mut client = Client::connect(&uri, NoTls)?;
        let mut transaction = client.transaction()?;
        let rows = transaction
            .query(
                &format!(
                    "SELECT reltuples::bigint AS estimate FROM pg_class WHERE oid ='\"{schema}\".\"{table}\"'::regclass"
                ),
                &[],
            )?;
        let count: i64 = rows[0].get(0);
        if count > 0 {
            logger.info(&format!(
                "Found approximately {} items in table \"{}\"",
                count, table,
            ));
        } else {
            logger.warn("Could not estimate table size");
        }
        // With portal we can execute a query and poll values from it in chunks
        let portal = transaction.bind(
            &format!("SELECT \"{pk}\"::text, \"{column}\" FROM \"{schema}\".\"{table}\";"),
            &[],
        )?;

        loop {
            // poll batch_size rows from portal and send it to embedding thread via channel
            let rows = transaction.query_portal(&portal, batch_size as i32)?;

            if rows.len() == 0 {
                break;
            }

            if tx.send(rows).is_err() {
                break;
            }
        }
        drop(tx);
        Ok(())
    });

    return Ok(handle);
}

// Embedding worker will listen to the producer channel
// and execute lantern_embeddings_core's corresponding function to generate embeddings
// we will here map each vector to it's row id (pk) before sending the results over channel
// So we will get Vec<Row<String, String> and output Vec<(String, Vec<f32>)> the output will
// contain generated embeddings for the text. If text will be null we will skip that row
fn embedding_worker(
    args: &cli::EmbeddingArgs,
    rx: Receiver<Vec<Row>>,
    tx: Sender<Vec<EmbeddingRecord>>,
    logger: Arc<Logger>,
) -> Result<JoinHandle<AnyhowVoidResult>, anyhow::Error> {
    let is_visual = args.visual.clone();
    let model = args.model.clone();
    let data_path = args.data_path.clone();
    let mut count: u64 = 0;

    let handle = std::thread::spawn(move || {
        let mut start = Instant::now();
        loop {
            let rows = rx.recv();
            if rows.is_err() {
                // channel has been closed
                break;
            }

            if count == 0 {
                // mark exact start time
                start = Instant::now();
            }

            let rows = rows.unwrap();
            let mut input_vectors: Vec<&str> = Vec::with_capacity(rows.len());
            let mut input_ids: Vec<String> = Vec::with_capacity(rows.len());

            for row in &rows {
                let col: Option<&str> = row.get(1);
                if col.is_some() {
                    input_vectors.push(col.unwrap());
                    input_ids.push(row.get::<usize, String>(0));
                }
            }

            if input_vectors.len() == 0 {
                continue;
            }

            let response_embeddings = if is_visual {
                clip::process_image(&model, &input_vectors, None, data_path.as_deref())
            } else {
                clip::process_text(&model, &input_vectors, None, data_path.as_deref())
            };

            if let Err(e) = response_embeddings {
                anyhow::bail!("{}", e);
            }

            let mut response_embeddings = response_embeddings.unwrap();

            count += response_embeddings.len() as u64;

            let duration = start.elapsed().as_secs();
            // avoid division by zero error
            let duration = if duration > 0 { duration } else { 1 };
            logger.debug(&format!(
                "Generated {} embeddings - speed {} emb/s",
                count,
                count / duration
            ));

            let mut response_data = Vec::with_capacity(rows.len());

            for _ in 0..response_embeddings.len() {
                response_data.push((input_ids.pop().unwrap(), response_embeddings.pop().unwrap()));
            }
            tx.send(response_data).unwrap();
        }
        logger.info("Embedding generation finished, waiting to export results...");
        drop(tx);
        Ok(())
    });

    return Ok(handle);
}

// DB exporter worker will create temp table with name _lantern_tmp_${rand(0,1000)}
// Then it will create writer stream which will COPY bytes from stdin to that table
// After that it will receiver the output embeddings mapped with row ids over the channel
// And write them using writer instance
// At the end we will flush the writer commit the transaction and UPDATE destination table
// Using our TEMP table data
fn db_exporter_worker(
    args: &cli::EmbeddingArgs,
    rx: Receiver<Vec<EmbeddingRecord>>,
    logger: Arc<Logger>,
) -> Result<JoinHandle<AnyhowVoidResult>, anyhow::Error> {
    let uri = args.out_uri.clone().unwrap_or(args.uri.clone());
    let pk = args.pk.clone();
    let column = args.out_column.clone();
    let table = args.out_table.clone().unwrap_or(args.table.clone());
    let schema = args.schema.clone();

    let handle = std::thread::spawn(move || {
        let mut client = Client::connect(&uri, NoTls)?;
        let mut transaction = client.transaction()?;
        let mut rng = rand::thread_rng();
        let temp_table_name = format!("_lantern_tmp_{}", rng.gen_range(0..1000));

        transaction
            .execute(
                &format!(
                    "CREATE TEMPORARY TABLE {temp_table_name} AS SELECT \"{pk}\", '{{}}'::REAL[] AS \"{column}\" FROM \"{schema}\".\"{table}\" LIMIT 0"
                ),
                &[],
            )?;
        transaction.commit()?;
        let mut transaction = client.transaction()?;
        let mut writer = transaction.copy_in(&format!("COPY {temp_table_name} FROM stdin"))?;
        let mut did_receive = false;
        loop {
            let rows = rx.recv();
            if rows.is_err() {
                // channel has been closed
                break;
            }

            if !did_receive {
                did_receive = true;
            }

            let rows = rows.unwrap();
            for row in &rows {
                writer.write(row.0.as_bytes())?;
                writer.write("\t".as_bytes())?;
                writer.write("{".as_bytes())?;
                let row_str: String = row.1.iter().map(|&x| x.to_string() + ",").collect();
                writer.write(row_str[0..row_str.len() - 1].as_bytes())?;
                drop(row_str);
                writer.write("}".as_bytes())?;
                writer.write("\n".as_bytes())?;
            }
            drop(rows);
        }

        if !did_receive {
            return Ok(());
        }

        writer.flush()?;
        writer.finish()?;
        transaction.execute(
            &format!(
                "ALTER TABLE \"{schema}\".\"{table}\" ADD COLUMN IF NOT EXISTS \"{column}\" REAL[]"
            ),
            &[],
        )?;
        transaction
            .execute(
                &format!(
                "UPDATE \"{schema}\".\"{table}\" dest SET \"{column}\" = src.\"{column}\" FROM \"{temp_table_name}\" src WHERE src.\"{pk}\" = dest.\"{pk}\""
            ),
                &[],
            )?;
        transaction.commit()?;
        logger.info(&format!(
            "Embeddings exported to table {} under column {}",
            &table, &column
        ));
        Ok(())
    });

    return Ok(handle);
}

fn csv_exporter_worker(
    args: &cli::EmbeddingArgs,
    rx: Receiver<Vec<EmbeddingRecord>>,
    logger: Arc<Logger>,
) -> Result<JoinHandle<AnyhowVoidResult>, anyhow::Error> {
    let csv_path = args.out_csv.clone().unwrap();
    let handle = std::thread::spawn(move || {
        let mut wtr = Writer::from_path(&csv_path).unwrap();

        loop {
            let rows = rx.recv();
            if rows.is_err() {
                // channel has been closed
                break;
            }
            let rows = rows.unwrap();
            for row in &rows {
                let vector_string = &format!(
                    "{{{}}}",
                    row.1
                        .iter()
                        .map(|f| f.to_string())
                        .collect::<Vec<String>>()
                        .join(",")
                );
                wtr.write_record(&[&row.0.to_string(), vector_string])
                    .unwrap();
            }
        }
        wtr.flush().unwrap();
        logger.info(&format!("Embeddings exported to {}", &csv_path));
        Ok(())
    });

    return Ok(handle);
}

pub fn create_embeddings_from_db(
    args: &cli::EmbeddingArgs,
    logger: Option<Logger>,
) -> Result<(), anyhow::Error> {
    let logger = Arc::new(logger.unwrap_or(Logger::new("Lantern Embeddings", LogLevel::Debug)));
    logger.info("Lantern CLI - Create Embeddings");
    logger.debug(&format!(
        "Model - {}, Visual - {}, Batch Size - {}",
        args.model, args.visual, args.batch_size
    ));

    // Exit process if any of the threads panic
    // This should never happen as thread results are handled below
    let process_hook_logger = logger.clone();
    panic::set_hook(Box::new(move |panic_info| {
        process_hook_logger.error(&format!("{}", panic_info));
        process::exit(1);
    }));

    // Create channel that will send the database rows to embedding worker
    let (producer_tx, producer_rx): (Sender<Vec<Row>>, Receiver<Vec<Row>>) = mpsc::channel();
    let (embedding_tx, embedding_rx): (
        Sender<Vec<EmbeddingRecord>>,
        Receiver<Vec<EmbeddingRecord>>,
    ) = mpsc::channel();

    // Create exporter based on provided args
    // For now we only have csv and db exporters
    let exporter_handle = if args.out_csv.is_some() {
        csv_exporter_worker(args, embedding_rx, logger.clone())?
    } else {
        db_exporter_worker(args, embedding_rx, logger.clone())?
    };

    // Collect the thread handles in a vector to wait them
    let handles = vec![
        producer_worker(args, producer_tx, logger.clone())?,
        embedding_worker(args, producer_rx, embedding_tx, logger.clone())?,
        exporter_handle,
    ];

    for handle in handles {
        if let Err(e) = handle.join().unwrap() {
            logger.error(&format!("{}", e));
            anyhow::bail!("{}", e);
        }
    }

    Ok(())
}

pub fn show_available_models(args: &cli::ShowModelsArgs) {
    let logger = Logger::new("Lantern Embeddings", LogLevel::Info);
    logger.info("Available Models\n");
    logger.print_raw(&clip::get_available_models(args.data_path.as_deref()));
}

use csv::Writer;
use lantern_embeddings_core::clip;
use rand::Rng;
use std::io::Write;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::thread::JoinHandle;
use std::time::Instant;

use postgres::{Client, NoTls, Row};

pub mod cli;

type EmbeddingRecord = (i32, Vec<f32>);

fn producer_worker(
    args: &cli::EmbeddingArgs,
    tx: Sender<Vec<Row>>,
) -> Result<JoinHandle<()>, anyhow::Error> {
    let uri = args.uri.clone();
    let pk = args.pk.clone();
    let column = args.column.clone();
    let table = args.table.clone();
    let batch_size = args.batch_size.clone();

    let handle = std::thread::spawn(move || {
        let mut client = Client::connect(&uri, NoTls).unwrap();
        let mut transaction = client.transaction().unwrap();
        let rows = transaction
            .query(
                &format!("SELECT COUNT(\"{}\") FROM \"{}\";", pk, table),
                &[],
            )
            .unwrap();
        let count: i64 = rows[0].get(0);
        println!("[*] Found {} items in table \"{}\"", count, table);
        // With portal we can execute a query and poll values from it in chunks
        let portal = transaction
            .bind(
                &format!("SELECT \"{}\", \"{}\" FROM \"{}\";", pk, column, table),
                &[],
            )
            .unwrap();

        loop {
            // poll batch_size rows from portal and send it to embedding thread via channel
            let rows = transaction
                .query_portal(&portal, batch_size as i32)
                .unwrap();

            if rows.len() == 0 {
                break;
            }

            if tx.send(rows).is_err() {
                break;
            }
        }
        drop(tx);
    });

    return Ok(handle);
}

fn embedding_worker(
    args: &cli::EmbeddingArgs,
    rx: Receiver<Vec<Row>>,
    tx: Sender<Vec<EmbeddingRecord>>,
) -> Result<JoinHandle<()>, anyhow::Error> {
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
            let mut input_vectors = Vec::with_capacity(rows.len());

            for row in &rows {
                let col: &str = row.get(1);
                input_vectors.push(col);
            }

            let response_embeddings = if is_visual {
                clip::process_image(&model, &input_vectors, None, data_path.as_deref())
            } else {
                clip::process_text(&model, &input_vectors, None, data_path.as_deref())
            };

            if let Err(e) = response_embeddings {
                eprintln!("{}", e);
                break;
            }

            let response_embeddings = response_embeddings.unwrap();

            count += response_embeddings.len() as u64;

            let duration = start.elapsed().as_secs();
            // avoid division by zero error
            let duration = if duration > 0 { duration } else { 1 };
            println!(
                "[*] Generated {} embeddings - speed {} emb/s",
                count,
                count / duration
            );

            let mut response_data = Vec::with_capacity(rows.len());

            for (i, embedding) in response_embeddings.iter().enumerate() {
                let id: i32 = rows[i].get(0);
                response_data.push((id, embedding.clone()));
            }
            tx.send(response_data).unwrap();
        }
        println!("[*] Embedding generation finished, waiting to export results...");
        drop(tx);
    });

    return Ok(handle);
}

fn db_exporter_worker(
    args: &cli::EmbeddingArgs,
    rx: Receiver<Vec<EmbeddingRecord>>,
) -> Result<JoinHandle<()>, anyhow::Error> {
    let uri = args.out_uri.clone().unwrap_or(args.uri.clone());
    let pk = args.pk.clone();
    let column = args.out_column.clone();
    let table = args.out_table.clone().unwrap_or(args.table.clone());

    let handle = std::thread::spawn(move || {
        let mut client = Client::connect(&uri, NoTls).unwrap();
        let mut transaction = client.transaction().unwrap();
        let mut rng = rand::thread_rng();
        let temp_table_name = format!("_lantern_tmp_{}", rng.gen_range(0..1000));

        transaction
            .execute(
                &format!(
                    "CREATE TEMPORARY TABLE {} (\"{}\" INT PRIMARY KEY, \"{}\" REAL[])",
                    &temp_table_name, &pk, &column
                ),
                &[],
            )
            .unwrap();
        transaction.commit().unwrap();
        let mut transaction = client.transaction().unwrap();
        let writer = transaction.copy_in(&format!("COPY {} FROM stdin", &temp_table_name));
        let mut writer = writer.unwrap();
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
                let row_str = format!("{}\t{}\n", row.0.to_string(), vector_string);
                writer.write_all(row_str.as_bytes()).unwrap();
            }
        }

        writer.flush().unwrap();
        writer.finish().unwrap();
        transaction
            .execute(
                &format!(
                "UPDATE {} dest SET \"{}\" = src.\"{}\" FROM {} src WHERE src.\"{}\" = dest.\"{}\"",
                &table, &column, &column, &temp_table_name, &pk, &pk
            ),
                &[],
            )
            .unwrap();
        transaction.commit().unwrap();
        println!(
            "[*] Embeddings exported to table {} under column {}",
            &table, &column
        );
    });

    return Ok(handle);
}

fn csv_exporter_worker(
    args: &cli::EmbeddingArgs,
    rx: Receiver<Vec<EmbeddingRecord>>,
) -> Result<JoinHandle<()>, anyhow::Error> {
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
        println!("[*] Embeddings exported to {}", &csv_path);
    });

    return Ok(handle);
}

pub fn create_embeddings_from_db(args: &cli::EmbeddingArgs) -> Result<(), anyhow::Error> {
    println!("[*] Lantern CLI - Create Embeddings");
    println!(
        "[*] Model - {}, Visual - {}, Batch Size - {}",
        args.model, args.visual, args.batch_size
    );
    let (producer_tx, producer_rx): (Sender<Vec<Row>>, Receiver<Vec<Row>>) = mpsc::channel();
    let (embedding_tx, embedding_rx): (
        Sender<Vec<EmbeddingRecord>>,
        Receiver<Vec<EmbeddingRecord>>,
    ) = mpsc::channel();

    let exporter_handle = if args.out_csv.is_some() {
        csv_exporter_worker(args, embedding_rx)?
    } else {
        db_exporter_worker(args, embedding_rx)?
    };

    let handles = vec![
        producer_worker(args, producer_tx)?,
        embedding_worker(args, producer_rx, embedding_tx)?,
        exporter_handle,
    ];

    for handle in handles {
        handle.join().unwrap();
    }

    Ok(())
}

pub fn show_available_models(args: &cli::ShowModelsArgs) {
    println!("[*] Lantern CLI - Available Models\n");
    println!("{}", clip::get_available_models(args.data_path.as_deref()));
}

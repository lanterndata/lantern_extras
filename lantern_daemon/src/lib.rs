use futures::{future, Future, StreamExt};
use lantern_embeddings::cli::EmbeddingArgs;
use lantern_logger::{LogLevel, Logger};
use std::ops::Deref;
use std::path::Path;
use std::pin::Pin;
use std::process;
use std::sync::Arc;
use tokio::fs;
use tokio_postgres::{AsyncMessage, Client, NoTls, Row};

use tokio::sync::{
    mpsc,
    mpsc::{Receiver, Sender},
};

pub mod cli;

#[derive(Debug)]
struct Job {
    id: String,
    db_uri: String,
    schema: String,
    table: String,
    column: String,
    out_column: String,
    model: String,
    batch_size: usize,
}

impl Job {
    pub fn new(row: Row) -> Job {
        Self {
            id: row.get::<&str, String>("id"),
            db_uri: row.get::<&str, String>("db_uri"),
            schema: row.get::<&str, String>("schema"),
            table: row.get::<&str, String>("table"),
            column: row.get::<&str, String>("column"),
            out_column: row.get::<&str, String>("out_column"),
            model: row.get::<&str, String>("model"),
            batch_size: 100,
        }
    }
}

type AnyhowVoidResult = Result<(), anyhow::Error>;
type VoidFuture = Pin<Box<dyn Future<Output = AnyhowVoidResult>>>;

fn get_full_table_name(schema: &str, table: &str) -> String {
    format!("\"{schema}\".\"{table}\"")
}

async fn db_notification_listener(
    db_uri: String,
    notification_channel: &'static str,
    queue_tx: Sender<String>,
    logger: Arc<Logger>,
) -> Result<(), anyhow::Error> {
    let (client, mut connection) = tokio_postgres::connect(&db_uri, tokio_postgres::NoTls).await?;

    let client = Arc::new(client);
    let client_ref = client.clone();
    // spawn new task to handle notifications
    let task = tokio::spawn(async move {
        let mut stream = futures::stream::poll_fn(move |cx| connection.poll_message(cx));
        logger.info("Lisening for notifications");

        while let Some(message) = stream.next().await {
            if let Err(e) = &message {
                logger.error(&format!("Failed to get message from db: {}", e));
            }

            let message = message.unwrap();

            if let AsyncMessage::Notification(not) = message {
                queue_tx.send(not.payload().to_owned()).await.unwrap();
            }
        }
        drop(client_ref);
    });

    client
        .batch_execute(&format!("LISTEN {notification_channel};"))
        .await?;

    task.await?;
    Ok(())
}

async fn embedding_worker(
    mut job_queue_rx: Receiver<Job>,
    client: Arc<Client>,
    schema: String,
    table: String,
    log_level: LogLevel,
    data_path: String,
    logger: Arc<Logger>,
) -> Result<(), anyhow::Error> {
    let schema = Arc::new(schema);
    let table = Arc::new(table);

    tokio::spawn(async move {
        logger.info("Embedding worker started");
        while let Some(job) = job_queue_rx.recv().await {
            logger.info(&format!("Starting execution of job {}", job.id));
            let client_ref = client.clone();
            let schema_ref = schema.clone();
            let table_ref = table.clone();
            let data_path = data_path.clone();

            let task_logger = Logger::new(&format!("Job {}", job.id), log_level.clone());
            let result = lantern_embeddings::create_embeddings_from_db(&EmbeddingArgs {
                pk: String::from("id"),
                model: job.model,
                schema: job.schema,
                uri: job.db_uri.clone(),
                out_uri: Some(job.db_uri),
                table: job.table.clone(),
                out_table: Some(job.table),
                column: job.column,
                out_column: job.out_column,
                batch_size: job.batch_size,
                data_path: Some(data_path),
                visual: false,
                out_csv: None,
            }, Some(task_logger));

            let full_table_name = get_full_table_name(schema_ref.deref(), table_ref.deref());
            if let Err(e) = result {
                // update failure reason
                client_ref.execute(&format!("UPDATE {full_table_name} SET init_failed_at=NOW(), updated_at=NOW(), init_failure_reason='{0}' WHERE id={1}", e.to_string(), job.id), &[]).await?;
            } else {
                // mark success
                client_ref.execute(&format!("UPDATE {full_table_name} SET init_finished_at=NOW(), updated_at=NOW() WHERE id={0}", job.id), &[]).await?;
            }


        }
        Ok(()) as AnyhowVoidResult
    })
    .await??;
    Ok(())
}

async fn startup_hook(
    client: Arc<Client>,
    table: &str,
    schema: &str,
    channel: &str,
    data_path: &str,
    logger: Arc<Logger>,
) -> Result<(), anyhow::Error> {
    logger.info("Setting up environment");
    // verify that table exists
    if let Err(_) = client
        .execute(
            &format!("SELECT ctid FROM \"{schema}\".\"{table}\" LIMIT 1"),
            &[],
        )
        .await
    {
        anyhow::bail!("Table {table} in schema {schema} does not exist");
    }

    // Set up trigger on table insert
    client
        .batch_execute(&format!(
            "
            CREATE OR REPLACE FUNCTION notify_lantern_daemon() RETURNS TRIGGER AS $$
              BEGIN
                PERFORM pg_notify('{channel}', NEW.id::TEXT);
                RETURN NULL;
              END;
            $$ LANGUAGE plpgsql;

            CREATE OR REPLACE TRIGGER trigger_lantern_jobs_insert
            AFTER INSERT
            ON {table}
            FOR EACH ROW
            EXECUTE PROCEDURE notify_lantern_daemon();
        ",
        ))
        .await?;

    let data_path = Path::new(data_path);
    if !data_path.exists() {
        fs::create_dir(data_path).await?;
    }
    Ok(())
}

async fn collect_pending_jobs(
    client: Arc<Client>,
    notification_tx: Sender<String>,
    table: String,
) -> AnyhowVoidResult {
    // Get all pending jobs and set them in queue
    let rows = client
        .query(
            &format!(
                "SELECT id::TEXT FROM {table} WHERE init_started_at IS NULL ORDER BY created_at"
            ),
            &[],
        )
        .await?;

    for row in rows {
        notification_tx.send(row.get::<usize, String>(0)).await?;
    }

    Ok(())
}

async fn job_queue_processor(
    client: Arc<Client>,
    mut notifications_rx: Receiver<String>,
    job_tx: Sender<Job>,
    schema: String,
    table: String,
) -> Result<(), anyhow::Error> {
    tokio::spawn(async move {
        while let Some(id) = notifications_rx.recv().await {
            let full_table_name =  get_full_table_name(&schema, &table);
            let updated_count = client.execute(&format!("UPDATE {0} SET init_started_at=NOW() WHERE init_started_at IS NULL AND id={id}", &full_table_name), &[]).await?;
            if updated_count == 0 {
                continue;
            }
            let row = client.query_one(&format!("SELECT id::TEXT, db_connection as db_uri, src_column as \"column\", dst_column as out_column, \"table\", \"schema\", embedding_model as model FROM {0} WHERE id={id}", &full_table_name), &[]).await?;
            job_tx.send(Job::new(row)).await?;
        }
        Ok(()) as AnyhowVoidResult
    })
    .await??;
    Ok(())
}

#[tokio::main]
pub async fn start(args: cli::DaemonArgs) -> Result<(), anyhow::Error> {
    let logger = Arc::new(Logger::new("Lantern Daemon", args.log_level.value()));
    logger.info("Staring Daemon");

    let (main_db_client, connection) = tokio_postgres::connect(&args.uri, NoTls).await?;

    tokio::spawn(async move { connection.await.unwrap() });

    let main_db_client = Arc::new(main_db_client);
    let notification_channel = "lantern_cloud_jobs";
    let data_path = "/usr/local/share/lantern-daemon";

    let (notification_queue_tx, notification_queue_rx): (Sender<String>, Receiver<String>) =
        mpsc::channel(args.queue_size);
    let (job_queue_tx, job_queue_rx): (Sender<Job>, Receiver<Job>) = mpsc::channel(args.queue_size);

    startup_hook(
        main_db_client.clone(),
        &args.table,
        &args.schema,
        &notification_channel,
        &data_path,
        logger.clone(),
    )
    .await?;

    let handles = vec![
        Box::pin(db_notification_listener(
            args.uri.clone(),
            &notification_channel,
            notification_queue_tx.clone(),
            logger.clone(),
        )) as VoidFuture,
        Box::pin(job_queue_processor(
            main_db_client.clone(),
            notification_queue_rx,
            job_queue_tx,
            args.schema.clone(),
            args.table.clone(),
        )) as VoidFuture,
        Box::pin(embedding_worker(
            job_queue_rx,
            main_db_client.clone(),
            args.schema.clone(),
            args.table.clone(),
            args.log_level.value(),
            data_path.to_owned(),
            logger.clone(),
        )) as VoidFuture,
        Box::pin(collect_pending_jobs(
            main_db_client.clone(),
            notification_queue_tx.clone(),
            args.table.clone(),
        )) as VoidFuture,
    ];

    if let Err(e) = future::try_join_all(handles).await {
        logger.error(&e.to_string());
        logger.error("Fatal error exiting process");
        process::exit(1);
    }

    Ok(())
}

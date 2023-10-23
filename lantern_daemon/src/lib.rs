pub mod cli;
mod client;
mod helpers;
mod types;

use client::toggle_client_job;
use futures::{future, StreamExt};
use helpers::{check_table_exists, get_full_table_name};
use lantern_embeddings::cli::EmbeddingArgs;
use lantern_logger::Logger;
use std::path::Path;
use std::process;
use std::sync::Arc;
use std::{ops::Deref, time::SystemTime};
use tokio::fs;
use tokio::sync::{
    mpsc,
    mpsc::{Receiver, Sender},
};
use tokio_postgres::{AsyncMessage, Client, NoTls};
use types::{AnyhowVoidResult, Job, JobInsertNotification, VoidFuture, JobUpdateNotification};

#[macro_use]
extern crate lazy_static;

async fn db_notification_listener(
    db_uri: String,
    notification_channel: &'static str,
    insert_queue_tx: Sender<JobInsertNotification>,
    update_queue_tx: Sender<JobUpdateNotification>,
    logger: Arc<Logger>,
) -> AnyhowVoidResult {
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
                let mut parts = not.payload().split(':');
                let action: &str = parts.next().unwrap();
                let id: &str = parts.next().unwrap();

                match action {
                    "insert" => {
                        insert_queue_tx
                            .send(JobInsertNotification {
                                id: id.to_owned(),
                                init: true,
                                filter: None,
                                limit: None,
                            })
                            .await
                            .unwrap();
                    }
                    "update" => {
                        update_queue_tx.send(JobUpdateNotification { id: id.to_owned(), generate_missing: true }).await.unwrap();
                    }
                    _ => logger.error(&format!("Invalid notification received {}", not.payload())),
                }
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
    notifications_tx: Sender<JobInsertNotification>,
    client: Arc<Client>,
    schema: String,
    table: String,
    data_path: String,
    logger: Arc<Logger>,
) -> AnyhowVoidResult {
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

            let task_logger = Logger::new(&format!("Job {}", job.id), logger.level.clone());
            let result = lantern_embeddings::create_embeddings_from_db(EmbeddingArgs {
                pk: String::from("id"),
                model: job.model,
                schema: job.schema.clone(),
                uri: job.db_uri.clone(),
                out_uri: Some(job.db_uri.clone()),
                table: job.table.clone(),
                out_table: Some(job.table.clone()),
                column: job.column.clone(),
                out_column: job.out_column.clone(),
                batch_size: job.batch_size,
                data_path: Some(data_path),
                visual: false,
                out_csv: None,
                filter: job.filter,
                limit: None
            }, Some(task_logger));

            if job.is_init {
                let full_table_name = get_full_table_name(schema_ref.deref(), table_ref.deref());
                if let Err(e) = result {
                    // update failure reason
                    client_ref.execute(&format!("UPDATE {full_table_name} SET init_failed_at=NOW(), updated_at=NOW(), init_failure_reason='{0}' WHERE id={1}", e.to_string(), job.id), &[]).await?;
                } else {
                    // mark success
                    client_ref.execute(&format!("UPDATE {full_table_name} SET init_finished_at=NOW(), updated_at=NOW() WHERE id={0}", job.id), &[]).await?;
                    toggle_client_job(job.id.clone(), job.db_uri.clone(), job.column.clone(), job.table.clone(), job.schema.clone(), logger.level.clone(), Some(notifications_tx.clone()), true ).await?;
                }
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
) -> AnyhowVoidResult {
    logger.info("Setting up environment");
    // verify that table exists
    let full_table_name = get_full_table_name(schema, table);
    check_table_exists(client.clone(), &full_table_name).await?;

    // Set up trigger on table insert
    client
        .batch_execute(&format!(
            "
            CREATE OR REPLACE FUNCTION notify_insert_lantern_daemon() RETURNS TRIGGER AS $$
              BEGIN
                PERFORM pg_notify('{channel}', 'insert:' || NEW.id::TEXT);
                RETURN NULL;
              END;
            $$ LANGUAGE plpgsql;

            CREATE OR REPLACE FUNCTION notify_update_lantern_daemon() RETURNS TRIGGER AS $$
              BEGIN
                IF (NEW.canceled_at IS NULL AND OLD.canceled_at IS NOT NULL) 
                OR (NEW.canceled_at IS NOT NULL AND OLD.canceled_at IS NULL)
                THEN
                     PERFORM pg_notify('{channel}', 'update:' || NEW.id::TEXT);
	            END IF;
                RETURN NEW;
              END;
            $$ LANGUAGE plpgsql;

            CREATE OR REPLACE TRIGGER trigger_lantern_jobs_insert
            AFTER INSERT 
            ON {full_table_name}
            FOR EACH ROW
            EXECUTE PROCEDURE notify_insert_lantern_daemon();

            CREATE OR REPLACE TRIGGER trigger_lantern_jobs_update
            AFTER UPDATE 
            ON {full_table_name}
            FOR EACH ROW
            EXECUTE PROCEDURE notify_update_lantern_daemon();
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
    insert_notification_tx: Sender<JobInsertNotification>,
    update_notification_tx: Sender<JobUpdateNotification>,
    table: String,
) -> AnyhowVoidResult {
    // Get all pending jobs and set them in queue
    let rows = client
        .query(
            &format!(
                "SELECT id::TEXT, canceled_at, init_finished_at, src_column, dst_column FROM {table} WHERE init_failed_at IS NULL ORDER BY created_at"
            ),
            &[],
        )
        .await?;

    for row in rows {
        let init_finished_at: Option<SystemTime> = row.get("init_finished_at");
        // TODO This can be optimized
        if init_finished_at.is_none() {
          insert_notification_tx.send(JobInsertNotification { id: row.get::<usize, String>(0).to_owned(), init: true, filter: None, limit: None }).await?;
        } else {
          update_notification_tx.send(JobUpdateNotification{ id: row.get::<usize, String>(0).to_owned(), generate_missing: true }).await?;
        }
    }

    Ok(())
}

async fn job_insert_processor(
    client: Arc<Client>,
    mut notifications_rx: Receiver<JobInsertNotification>,
    job_tx: Sender<Job>,
    schema: String,
    table: String,
    logger: Arc<Logger>,
) -> AnyhowVoidResult {
    tokio::spawn(async move {
        while let Some(notification) = notifications_rx.recv().await {
            let full_table_name =  get_full_table_name(&schema, &table);
            let id = notification.id;
            if notification.init {
                // Only update init time if this is the first time job is being executed
                let updated_count = client.execute(&format!("UPDATE {0} SET init_started_at=NOW() WHERE init_started_at IS NULL AND id={id}", &full_table_name), &[]).await?;
                if updated_count == 0 {
                    continue;
                }
            }
            
            let job_result = client.query_one(&format!("SELECT id::TEXT, db_connection as db_uri, src_column as \"column\", dst_column, \"table\", \"schema\", embedding_model as model FROM {0} WHERE id={id}", &full_table_name), &[]).await;

            if let Ok(row) = job_result {
                let mut job = Job::new(row);
                job.set_is_init(notification.init);
                if notification.filter.is_some() {
                  job.set_filter(&notification.filter.unwrap());
                }
                job_tx.send(job).await?;
            } else {
                logger.error(&format!("Error while getting job {id}: {}", job_result.err().unwrap()));
            }
        }
        Ok(()) as AnyhowVoidResult
    })
    .await??;
    Ok(())
}

async fn job_update_processor(
    client: Arc<Client>,
    mut update_queue_rx: Receiver<JobUpdateNotification>,
    job_insert_queue_tx: Sender<JobInsertNotification>,
    schema: String,
    table: String,
    logger: Arc<Logger>,
) -> AnyhowVoidResult {
    tokio::spawn(async move {
        while let Some(notification) = update_queue_rx.recv().await {
            let full_table_name =  get_full_table_name(&schema, &table);
            let id = notification.id;
            let row = client.query_one(&format!("SELECT db_connection as db_uri, dst_column, src_column as \"column\", \"table\", \"schema\", canceled_at FROM {0} WHERE id={id}", &full_table_name), &[]).await?;
            let src_column = row.get::<&str, String>("column").to_owned();
            let out_column = row.get::<&str, String>("dst_column").to_owned();

            let canceled_at: Option<SystemTime> = row.get("canceled_at");
            logger.debug(&format!("Update job {id}: is_canceled: {}", canceled_at.is_some()));
            toggle_client_job(id.clone(), row.get::<&str, String>("db_uri").to_owned(), row.get::<&str, String>("column").to_owned(), row.get::<&str, String>("table").to_owned(), row.get::<&str, String>("schema").to_owned(), logger.level.clone(), Some(job_insert_queue_tx.clone()), canceled_at.is_none()).await?;

            if canceled_at.is_none() && notification.generate_missing {
                // this will be on startup to generate embeddings for rows that might be inserted
                // while daemon is offline
                job_insert_queue_tx.send(JobInsertNotification { id: id.clone(), init: false, filter: Some(format!("\"{src_column}\" IS NOT NULL AND \"{out_column}\" IS NULL")), limit: None }).await?;
            }
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

    let (insert_notification_queue_tx, insert_notification_queue_rx): (
        Sender<JobInsertNotification>,
        Receiver<JobInsertNotification>,
    ) = mpsc::channel(args.queue_size);
    let (update_notification_queue_tx, update_notification_queue_rx): (
        Sender<JobUpdateNotification>,
        Receiver<JobUpdateNotification>,
    ) = mpsc::channel(args.queue_size);
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
            insert_notification_queue_tx.clone(),
            update_notification_queue_tx.clone(),
            logger.clone(),
        )) as VoidFuture,
        Box::pin(job_insert_processor(
            main_db_client.clone(),
            insert_notification_queue_rx,
            job_queue_tx,
            args.schema.clone(),
            args.table.clone(),
            logger.clone(),
        )) as VoidFuture,
        Box::pin(job_update_processor(
            main_db_client.clone(),
            update_notification_queue_rx,
            insert_notification_queue_tx.clone(),
            args.schema.clone(),
            args.table.clone(),
            logger.clone(),
        )) as VoidFuture,
        Box::pin(embedding_worker(
            job_queue_rx,
            insert_notification_queue_tx.clone(),
            main_db_client.clone(),
            args.schema.clone(),
            args.table.clone(),
            data_path.to_owned(),
            logger.clone(),
        )) as VoidFuture,
        Box::pin(collect_pending_jobs(
            main_db_client.clone(),
            insert_notification_queue_tx.clone(),
            update_notification_queue_tx.clone(),
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

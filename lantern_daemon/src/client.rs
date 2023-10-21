use crate::helpers::{check_table_exists, get_full_table_name};
use crate::types::{AnyhowVoidResult, JobInsertNotification, VoidFuture};
use futures::{future, StreamExt};
use lantern_logger::{LogLevel, Logger};
use std::sync::Arc;
use tokio::sync::{
    mpsc,
    mpsc::{Receiver, Sender},
};
use tokio_postgres::{AsyncMessage, Client, NoTls};

// static hashmap client_jobs: <id, tx>

pub async fn toggle_client_job(
    job_id: String,
    db_uri: String,
    src_column: String,
    table: String,
    schema: String,
    log_level: LogLevel,
    job_insert_queue_tx: Option<Sender<JobInsertNotification>>,
    enable: bool,
) -> AnyhowVoidResult {
    if enable {
        let job_insert_queue_tx = job_insert_queue_tx.unwrap();
        tokio::spawn(async move {
            let _ = start_client_job(
                job_id,
                db_uri,
                src_column,
                table,
                schema,
                job_insert_queue_tx,
                log_level,
            )
            .await;
        });
    } else {
        println!("STOP");
        // stop_client_job
    }

    Ok(())
}

pub async fn setup_client_triggers(
    client: Arc<Client>,
    column: &str,
    table: &str,
    schema: &str,
    channel: Arc<String>,
    job_id: &str,
    logger: Arc<Logger>,
) -> AnyhowVoidResult {
    logger.info("Setting Up Client Triggers");
    // verify that table exists
    let full_table_name = get_full_table_name(schema, table);
    check_table_exists(client.clone(), &full_table_name).await?;

    // Set up trigger on table insert
    client
        .batch_execute(&format!(
            "
            CREATE OR REPLACE FUNCTION notify_insert_lantern_daemon_{table}_{column}() RETURNS TRIGGER AS $$
              BEGIN
                PERFORM pg_notify('{channel}', NEW.id::TEXT || ':' || '{job_id}');
                RETURN NULL;
              END;
            $$ LANGUAGE plpgsql;

            CREATE OR REPLACE TRIGGER trigger_lantern_jobs_insert_{column}
            AFTER INSERT 
            ON {full_table_name}
            FOR EACH ROW
            EXECUTE PROCEDURE notify_insert_lantern_daemon_{table}_{column}();
        ",
        ))
        .await?;

    Ok(())
}

async fn client_notification_listener(
    db_uri: Arc<String>,
    notification_channel: Arc<String>,
    job_insert_queue_tx: Sender<JobInsertNotification>,
    logger: Arc<Logger>,
) -> AnyhowVoidResult {
    let (client, mut connection) = tokio_postgres::connect(&db_uri, NoTls).await?;

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
                let pk: &str = parts.next().unwrap();
                let job_id: &str = parts.next().unwrap();
                // TODO take from job
                let pk_name = "id";

                let result = job_insert_queue_tx
                    .send(JobInsertNotification {
                        id: job_id.to_owned(),
                        init: false,
                        filter: Some(format!("\"{pk_name}\"={pk}")),
                        limit: None,
                    })
                    .await;

                if let Err(e) = result {
                    logger.error(&e.to_string());
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

pub async fn start_client_job(
    job_id: String,
    db_uri: String,
    src_column: String,
    table: String,
    schema: String,
    job_insert_queue_tx: Sender<JobInsertNotification>,
    log_level: LogLevel,
) -> AnyhowVoidResult {
    let logger = Arc::new(Logger::new(&format!("Job {job_id}"), log_level));
    logger.info("Staring Client Listener");

    let (db_client, connection) = tokio_postgres::connect(&db_uri, NoTls).await?;
    let db_client = Arc::new(db_client);
    let db_connection_logger = logger.clone();
    let db_connection_task = tokio::spawn(async move {
        if let Err(e) = connection.await {
            db_connection_logger.error(&e.to_string());
        }
    });
    let notification_channel =
        Arc::new(format!("lantern_client_notifications_{table}_{src_column}"));
    let db_uri = Arc::new(db_uri);
    setup_client_triggers(
        db_client,
        &src_column,
        &table,
        &schema,
        notification_channel.clone(),
        &job_id,
        logger.clone(),
    )
    .await?;
    db_connection_task.abort();

    client_notification_listener(
        db_uri.clone(),
        notification_channel.clone(),
        job_insert_queue_tx.clone(),
        logger.clone(),
    )
    .await?;
    // this function should return channel producer which when received will kill the job calling
    // stop_client_job() function
    // if let Err(e) = future::try_join_all(handles).await {
    //     logger.error(&e.to_string());
    //     // TODO handle cient error
    // }
    Ok(())
}

pub async fn stop_client_job() -> AnyhowVoidResult {
    // remove triggers
    // remove database connection
    // remove job from hashmap
    Ok(())
}

use super::client_embedding_jobs::{stop_client_job, toggle_client_job};
use super::helpers::{
    cancellation_handler, db_notification_listener, get_missing_rows_filter, notify_job,
    remove_job_handle, schedule_job_retry, set_job_handle, startup_hook,
};
use super::types::{
    ClientJobsMap, EmbeddingJob, JobBatchingHashMap, JobEvent, JobEventHandlersMap,
    JobInsertNotification, JobLabelsMap, JobRunArgs, JobUpdateNotification,
};
use crate::daemon::helpers::anyhow_wrap_connection;
use crate::embeddings::cli::EmbeddingArgs;
use crate::logger::Logger;
use crate::utils::{get_common_embedding_ignore_filters, get_full_table_name, quote_ident};
use crate::{embeddings, types::*};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::time::SystemTime;
use tokio::fs;
use tokio::sync::mpsc::{Receiver, Sender, UnboundedReceiver, UnboundedSender};
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio_postgres::types::ToSql;
use tokio_postgres::{Client, NoTls};
use tokio_util::sync::CancellationToken;

pub const JOB_TABLE_DEFINITION: &'static str = r#"
"id" SERIAL PRIMARY KEY,
"schema" text NOT NULL DEFAULT 'public',
"table" text NOT NULL,
"pk" text NOT NULL DEFAULT 'id',
"label" text NULL,
"runtime" text NOT NULL DEFAULT 'ort',
"runtime_params" jsonb,
"src_column" text NOT NULL,
"dst_column" text NOT NULL,
"embedding_model" text NOT NULL,
"created_at" timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
"updated_at" timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
"canceled_at" timestamp,
"init_started_at" timestamp,
"init_finished_at" timestamp,
"init_failed_at" timestamp,
"init_failure_reason" text,
"init_progress" int2 DEFAULT 0
"#;

pub const USAGE_TABLE_DEFINITION: &'static str = r#"
"id" SERIAL PRIMARY KEY,
"job_id" INT NOT NULL,
"rows" INT NOT NULL,
"tokens" INT NOT NULL,
"failed" BOOL NOT NULL DEFAULT FALSE,
"created_at" timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP
"#;

const EMB_USAGE_TABLE_NAME: &'static str = "embedding_usage_info";
const EMB_LOCK_TABLE_NAME: &'static str = "_lantern_emb_job_locks";

async fn lock_row(
    client: Arc<Client>,
    lock_table_name: &str,
    logger: Arc<Logger>,
    job_id: i32,
    row_id: &str,
) -> bool {
    let res = client
        .execute(
            &format!("INSERT INTO {lock_table_name} (job_id, row_id) VALUES ($1, $2)"),
            &[&job_id, &row_id],
        )
        .await;

    if let Err(e) = res {
        if !e.to_string().to_lowercase().contains("duplicate") {
            logger.error(&format!(
                "Error while locking row: {row_id} for job: {job_id} : {e}"
            ));
        }
        return false;
    }
    true
}

#[allow(dead_code)]
async fn lock_rows(
    client: Arc<Client>,
    lock_table_name: &str,
    logger: Arc<Logger>,
    job_id: i32,
    row_ids: &Vec<String>,
) -> bool {
    let mut statement = format!("INSERT INTO {lock_table_name} (job_id, row_id) VALUES ");

    let mut idx = 0;
    let values: Vec<&(dyn ToSql + Sync)> = row_ids
        .iter()
        .flat_map(|row_id| {
            let comma = if idx == row_ids.len() - 1 { "" } else { "," };
            statement = format!(
                "{statement}({job_id}, ${param_num}){comma}",
                param_num = idx + 1
            );
            idx += 1;
            [row_id as &(dyn ToSql + Sync)]
        })
        .collect();

    let res = client.execute(&statement, &values[..]).await;

    if let Err(e) = res {
        if !e.to_string().to_lowercase().contains("duplicate") {
            logger.error(&format!(
                "Error while locking rows: {:?} for job: {job_id} : {e}",
                row_ids
            ));
        }
        return false;
    }
    true
}

async fn unlock_rows(
    client: Arc<Client>,
    lock_table_name: &str,
    logger: Arc<Logger>,
    job_id: i32,
    row_ids: &Vec<String>,
) {
    let mut row_ids_query = "".to_owned();
    let mut params: Vec<&(dyn ToSql + Sync)> = row_ids
        .iter()
        .enumerate()
        .map(|(idx, id)| {
            let comma = if idx < row_ids.len() - 1 { "," } else { "" };
            row_ids_query = format!("{row_ids_query}${}{comma}", idx + 1);
            id as &(dyn ToSql + Sync)
        })
        .collect();
    params.push(&job_id);
    let res = client
        .execute(
            &format!("DELETE FROM {lock_table_name} WHERE job_id=${job_id_pos} AND row_id IN ({row_ids_query})", job_id_pos=params.len()),
            &params,
        )
        .await;

    if let Err(e) = res {
        logger.error(&format!(
            "Error while unlocking rows: {:?} for job: {job_id} : {e}",
            row_ids
        ));
    }
}

async fn stream_job(
    logger: Arc<Logger>,
    client_jobs_map: Arc<ClientJobsMap>,
    main_client: Arc<Client>,
    job_queue_tx: Sender<EmbeddingJob>,
    notifications_tx: UnboundedSender<JobInsertNotification>,
    jobs_table_name: String,
    _lock_table_name: String,
    jobs_map: Arc<JobEventHandlersMap>,
    job: EmbeddingJob,
) -> AnyhowVoidResult {
    let top_logger = logger.clone();
    let job_id = job.id;
    let notifications_tx_clone = notifications_tx.clone();
    let job_clone = job.clone();
    let jobs_map_clone = jobs_map.clone();
    let client_jobs_map_clone = client_jobs_map.clone();
    let jobs_table_name_clone = jobs_table_name.clone();

    let task = tokio::spawn(async move {
        logger.info(&format!("Start streaming job {}", job.id));
        // Enable triggers for job
        if job.is_init {
            let client_jobs_map = client_jobs_map.clone();
            toggle_client_job(
                client_jobs_map,
                job.id.clone(),
                job.db_uri.clone(),
                job.pk.clone(),
                job.column.clone(),
                job.out_column.clone(),
                job.table.clone(),
                job.schema.clone(),
                logger.level.clone(),
                &logger.label,
                Some(notifications_tx.clone()),
                true,
            )
            .await?;
        }

        let (job_event_tx, mut job_event_rx): (Sender<JobEvent>, Receiver<JobEvent>) =
            mpsc::channel(1);
        set_job_handle(&jobs_map, job.id, job_event_tx).await?;

        let connection_logger = logger.clone();

        let (mut job_client, connection) = tokio_postgres::connect(&job.db_uri, NoTls).await?;
        let job_id = job.id;

        let job_clone = job.clone();
        let notifications_tx_clone = notifications_tx.clone();
        tokio::spawn(async move {
            if let Err(e) = connection.await {
                connection_logger.error(&format!("Error while streaming job: {job_id}. {e}"));
                if job.is_init {
                    toggle_client_job(
                        client_jobs_map.clone(),
                        job_clone.id,
                        job_clone.db_uri,
                        job_clone.pk,
                        job_clone.column,
                        job_clone.out_column,
                        job_clone.table,
                        job_clone.schema,
                        connection_logger.level.clone(),
                        &connection_logger.label,
                        Some(notifications_tx_clone),
                        false,
                    )
                    .await?;
                }
            }
            Ok::<(), anyhow::Error>(())
        });

        let column = &job.column;
        let out_column = &job.out_column;
        let schema = &job.schema;
        let table = &job.table;
        let full_table_name = get_full_table_name(schema, table);

        let filter_sql = format!(
            "WHERE {out_column} IS NULL AND {common_filter}",
            common_filter = get_common_embedding_ignore_filters(&quote_ident(column)),
        );

        let transaction = job_client.transaction().await?;

        transaction
            .execute("SET idle_in_transaction_session_timeout=3600000", &[])
            .await?;
        let total_rows = transaction
            .query_one(
                &format!("SELECT COUNT(*) FROM {full_table_name} {filter_sql};"),
                &[],
            )
            .await?;
        let total_rows: i64 = total_rows.get(0);
        let total_rows = total_rows as usize;

        if total_rows == 0 {
            if job.is_init {
                transaction.execute(
                  &format!(
                      "UPDATE {jobs_table_name} SET init_finished_at=NOW(), updated_at=NOW(), init_progress=100 WHERE id=$1"
                  ),
                  &[&job.id],
                )
                .await?;
                transaction.commit().await?;
            }

            return Ok(());
        }

        let portal = transaction
            .bind(
                &format!(
                    "SELECT {pk}::text FROM {full_table_name} {filter_sql};",
                    pk = quote_ident(&job.pk)
                ),
                &[],
            )
            .await?;
        let mut progress = 0;
        let mut processed_rows = 0;

        let batch_size = embeddings::get_default_batch_size(&job.model) as i32;
        loop {
            // poll batch_size rows from portal and send it to embedding thread via channel
            let rows = transaction.query_portal(&portal, batch_size).await?;

            if rows.len() == 0 {
                break;
            }

            let mut streamed_job = job.clone();
            let row_ids: Vec<String> = rows.iter().map(|r| r.get::<usize, String>(0)).collect();
            // If this is not init job, but startup job to generate missing rows
            // We will lock the row ids, so if 2 daemons will be started at the same time
            // Same rows won't be processed by both of them
            // TODO:: This check is now commented because
            // There might be case when the job will be stopped abnoarmally and
            // Rows will stay locked, so in next run the missed rows will be considered as locked
            // And won't be taken to be processed
            // streamed_job.set_row_ids(row_ids);
            // if !job.is_init
            //     && !lock_rows(
            //         main_client.clone(),
            //         &lock_table_name,
            //         logger.clone(),
            //         job_id,
            //         streamed_job.row_ids.as_ref().unwrap(),
            //     )
            //     .await
            // {
            //     logger.warn("Another daemon instance is already processing job {job_id}. Exitting streaming loop");
            //     break;
            // }

            streamed_job.set_id_filter(&row_ids);

            processed_rows += row_ids.len();
            if job.is_init {
                let new_progress = ((processed_rows as f32 / total_rows as f32) * 100.0) as u8;

                if new_progress > progress {
                    progress = new_progress;
                    streamed_job.set_report_progress(progress);
                }
            }

            if processed_rows == total_rows {
                streamed_job.set_is_last_chunk(true);
            }

            job_queue_tx.send(streamed_job).await?;

            // Wait for response from embedding worker
            if let Some(msg) = job_event_rx.recv().await {
                if let JobEvent::Errored(err) = msg {
                    anyhow::bail!(err);
                }
            }
        }

        Ok(())
    });

    if let Err(e) = task.await? {
        top_logger.error(&format!("Error while streaming job {job_id}: {e}"));
        remove_job_handle(&jobs_map_clone, job_id).await?;
        if job_clone.is_init {
            main_client.execute(&format!("UPDATE {jobs_table_name_clone} SET init_failed_at=NOW(), updated_at=NOW(), init_failure_reason=$1 WHERE id=$2 AND init_finished_at IS NULL"), &[&e.to_string(), &job_id]).await?;
            toggle_client_job(
                client_jobs_map_clone.clone(),
                job_clone.id,
                job_clone.db_uri,
                job_clone.pk,
                job_clone.column,
                job_clone.out_column,
                job_clone.table,
                job_clone.schema,
                top_logger.level.clone(),
                &top_logger.label,
                Some(notifications_tx_clone),
                false,
            )
            .await?;
        }
    }

    Ok(())
}

async fn embedding_worker(
    mut job_queue_rx: Receiver<EmbeddingJob>,
    job_queue_tx: Sender<EmbeddingJob>,
    db_uri: String,
    schema: String,
    table: String,
    jobs_map: Arc<JobEventHandlersMap>,
    logger: Arc<Logger>,
) -> AnyhowVoidResult {
    let schema = Arc::new(schema);
    let table = Arc::new(table);
    let jobs_table_name = Arc::new(get_full_table_name(&schema, &table));
    let usage_table_name = get_full_table_name(&schema, EMB_USAGE_TABLE_NAME);

    tokio::spawn(async move {
        let (client, connection) = tokio_postgres::connect(&db_uri, NoTls).await?;
        tokio::spawn(async move { connection.await });
        let client = Arc::new(client);
        logger.info("Embedding worker started");
        while let Some(job) = job_queue_rx.recv().await {
            let client_ref = client.clone();
            let orig_job_clone = job.clone();
            let job = Arc::new(job);

            let task_logger = Logger::new(
                &format!("{}|{}|{:?}", logger.label, job.id, job.runtime),
                logger.level.clone(),
            );
            let job_clone = job.clone();

            let result = crate::embeddings::create_embeddings_from_db(
                EmbeddingArgs {
                    pk: job.pk.clone(),
                    model: job_clone.model.clone(),
                    schema: job_clone.schema.clone(),
                    uri: job_clone.db_uri.clone(),
                    out_uri: Some(job_clone.db_uri.clone()),
                    table: job_clone.table.clone(),
                    out_table: Some(job_clone.table.clone()),
                    column: job_clone.column.clone(),
                    out_column: job_clone.out_column.clone(),
                    batch_size: job_clone.batch_size,
                    runtime: job_clone.runtime.clone(),
                    runtime_params: job_clone.runtime_params.clone(),
                    visual: false,
                    stream: true,
                    create_column: false,
                    out_csv: None,
                    filter: job_clone.filter.clone(),
                    limit: None,
                },
                false,
                None,
                None,
                Some(task_logger),
            );

            match result {
                Ok((processed_rows, processed_tokens)) => {
                    if processed_tokens > 0 {
                        let res = client_ref
                            .execute(
                                &format!(
                                    "INSERT INTO {usage_table_name} (job_id, rows, tokens) VALUES ($1, $2, $3)",
                                ),
                                &[&job.id, &(processed_rows as i32), &(processed_tokens as i32)],
                            )
                            .await;

                        if let Err(e) = res {
                            logger.error(&format!(
                                "Error while inserting usage info for job {job_id}: {e}",
                                job_id = job.id
                            ));
                        }
                    }

                    if let Some(progress) = job.report_progress {
                        let update_statement = if job.is_last_chunk {
                            "init_finished_at=NOW(), updated_at=NOW(), init_progress=100".to_owned()
                        } else {
                            format!("init_progress={progress}")
                        };

                        client_ref
                            .execute(
                                &format!(
                                    "UPDATE {jobs_table_name} SET {update_statement} WHERE id=$1"
                                ),
                                &[&job.id],
                            )
                            .await?;
                    }

                    // Send done event via channel
                    notify_job(jobs_map.clone(), job.id, JobEvent::Done).await;

                    if job.is_last_chunk {
                        remove_job_handle(&jobs_map, job.id).await?;
                    }
                }
                Err(e) => {
                    logger.error(&format!(
                        "Error while executing job {job_id} (is_init: {is_init}): {e}",
                        job_id = job.id,
                        is_init = job.is_init
                    ));

                    if !job.is_init {
                        schedule_job_retry(
                            logger.clone(),
                            orig_job_clone,
                            job_queue_tx.clone(),
                            Duration::from_secs(300),
                        )
                        .await;


                        let row_count = if job.row_ids.is_some() {
                            job.row_ids.as_ref().unwrap().len()
                        } else {
                            0
                        };
                        let res = client_ref
                            .execute(
                                &format!(
                                    "INSERT INTO {usage_table_name} (job_id, rows, tokens, failed) VALUES ($1, $2, 0, TRUE)",
                                ),
                                &[&job.id, &(row_count as i32)],
                            )
                            .await;


                        if let Err(e) = res {
                            logger.error(&format!(
                                "Error while updating failures for {job_id}: {e}",
                                job_id = job.id
                            ));
                        }
                    }

                    // Send error via channel, so init streaming task will catch that
                    notify_job(jobs_map.clone(), job.id, JobEvent::Errored(e.to_string())).await;
                }
            }

            if let Some(row_ids) = &job.row_ids {
                // If this is a job triggered from notification (new row inserted or row was updated)
                // Then we need to remove the entries from lock table for this rows
                // As we are using table ctid for lock key, after table VACUUM the ctids may repeat
                // And if new row will be inserted with previously locked ctid
                // it won't be taken by daemon
                let lock_table_name = get_full_table_name(&schema, EMB_LOCK_TABLE_NAME);
                unlock_rows(
                    client_ref,
                    &lock_table_name,
                    logger.clone(),
                    job.id,
                    row_ids,
                )
                .await;
            }
        }
        Ok(()) as AnyhowVoidResult
    })
    .await??;
    Ok(())
}

async fn collect_pending_jobs(
    client: Arc<Client>,
    update_notification_tx: UnboundedSender<JobUpdateNotification>,
    table: String,
) -> AnyhowVoidResult {
    // Get all pending jobs and set them in queue
    let rows = client
        .query(
            &format!("SELECT id FROM {table} WHERE init_failed_at IS NULL and canceled_at IS NULL ORDER BY id"),
            &[],
        )
        .await?;

    for row in rows {
        // TODO This can be optimized
        update_notification_tx.send(JobUpdateNotification {
            id: row.get::<usize, i32>(0).to_owned(),
            generate_missing: true,
        })?;
    }

    Ok(())
}

async fn job_insert_processor(
    mut notifications_rx: UnboundedReceiver<JobInsertNotification>,
    notifications_tx: UnboundedSender<JobInsertNotification>,
    job_tx: Sender<EmbeddingJob>,
    db_uri: String,
    schema: String,
    table: String,
    daemon_label: String,
    data_path: &'static str,
    jobs_map: Arc<JobEventHandlersMap>,
    job_batching_hashmap: Arc<JobBatchingHashMap>,
    client_jobs_map: Arc<ClientJobsMap>,
    job_labels_map: Arc<JobLabelsMap>,
    logger: Arc<Logger>,
) -> AnyhowVoidResult {
    // This function will have 2 running tasks
    // The first task will receive notification which can come from 4 sources
    // 1 - a new job is inserted into jobs table
    // 2 - after starting daemon job table scan was happened and we should check for any inserts we
    // missed while daemon was offline
    // 3 - job's canceled_at was changed to NULL, which means we should reactivate the job and
    //   generate embeddings for the rows which were inserted while job had canceled_at
    // 4 - a new row was inserted into client database and we should generate an embedding for that
    //   row
    // 1-3 cases works similiar we just send a new job record to embedding worker receiver which
    // will start generating embeddings for that job
    // for the 4th case we will lock the row (insert a record in our lock table) and set that row
    // in a hashmap item for that job (e.g { [job_id] => Vec<row_id1, row_id2> })
    // each 10 sconds the second running task the collector task will drain the hashmap and create
    // batch jobs for the rows. This will optimize embedding generation as if there will be lots of
    // inserts to the table between 10 seconds all that rows will be batched.
    let full_table_name = Arc::new(get_full_table_name(&schema, &table));
    let job_query_sql = Arc::new(format!("SELECT id, pk, label, src_column as \"column\", dst_column, \"table\", \"schema\", embedding_model as model, runtime, runtime_params::text, init_finished_at FROM {0}", &full_table_name));

    let db_uri_r1 = db_uri.clone();
    let full_table_name_r1 = full_table_name.clone();
    let job_query_sql_r1 = job_query_sql.clone();
    let job_tx_r1 = job_tx.clone();
    let logger_r1 = logger.clone();
    let lock_table_name = Arc::new(get_full_table_name(&schema, EMB_LOCK_TABLE_NAME));
    let job_batching_hashmap_r1 = job_batching_hashmap.clone();
    let job_labels_map_r1 = job_labels_map.clone();

    let insert_processor_task = tokio::spawn(async move {
        let (insert_client, connection) = tokio_postgres::connect(&db_uri_r1, NoTls).await?;
        let insert_client = Arc::new(insert_client);
        tokio::spawn(async move { connection.await });
        while let Some(notification) = notifications_rx.recv().await {
            let id = notification.id;

            // check if job's label is not matching the label of current daemon instance
            // do not process the row
            if let Some(label) = job_labels_map.read().await.get(&id) {
                if label != &daemon_label {
                    continue;
                }
            }

            if let Some(row_id) = notification.row_id {
                // Do this in a non-blocking way to not block collecting of updates while locking
                let client_r1 = insert_client.clone();
                let logger_r1 = logger_r1.clone();
                let job_batching_hashmap_r1 = job_batching_hashmap_r1.clone();
                let lock_table_name = lock_table_name.clone();
                tokio::spawn(async move {
                    // Single row update received from client job, lock row and add to batching map
                    let status = lock_row(
                        client_r1.clone(),
                        &lock_table_name,
                        logger_r1.clone(),
                        id,
                        &row_id,
                    )
                    .await;

                    if status {
                        // this means locking was successfull and row will be processed
                        // from this daemon
                        let mut jobs = job_batching_hashmap_r1.lock().await;
                        let job = jobs.get_mut(&id);

                        if let Some(job_vec) = job {
                            job_vec.push(row_id.to_owned());
                        } else {
                            jobs.insert(id, vec![row_id.to_owned()]);
                        }

                        drop(jobs);
                    }
                });

                continue;
            }

            let job_result = insert_client
                .query_one(
                    &format!("{job_query_sql_r1} WHERE id=$1 AND canceled_at IS NULL"),
                    &[&id],
                )
                .await;

            if let Ok(row) = job_result {
                let is_init = row
                    .get::<&str, Option<SystemTime>>("init_finished_at")
                    .is_none();

                let job = EmbeddingJob::new(row, data_path, &db_uri_r1);

                if let Err(e) = &job {
                    logger_r1.error(&format!("Error while creating job {id}: {e}",));
                    continue;
                }

                let mut job = job.unwrap();

                if let Some(label) = &job.label {
                    // insert label in cache
                    job_labels_map.write().await.insert(job.id, label.clone());

                    if label != &daemon_label {
                        continue;
                    }
                }

                if is_init {
                    // Only update init time if this is the first time job is being executed
                    let updated_count = insert_client.execute(&format!("UPDATE {0} SET init_started_at=NOW() WHERE init_started_at IS NULL AND id=$1", &full_table_name_r1), &[&id]).await?;
                    if updated_count == 0 && !notification.generate_missing {
                        continue;
                    }
                }

                job.set_is_init(is_init);
                if let Some(filter) = notification.filter {
                    job.set_filter(&filter);
                }
                // TODO:: Check if passing insert_client does not make deadlocks
                tokio::spawn(stream_job(
                    logger_r1.clone(),
                    client_jobs_map.clone(),
                    insert_client.clone(),
                    job_tx_r1.clone(),
                    notifications_tx.clone(),
                    full_table_name_r1.to_string(),
                    lock_table_name.to_string(),
                    jobs_map.clone(),
                    job,
                ));
            } else {
                logger_r1.error(&format!(
                    "Error while getting job {id}: {}",
                    job_result.err().unwrap()
                ));
            }
        }
        Ok(()) as AnyhowVoidResult
    });

    let job_batching_hashmap = job_batching_hashmap.clone();
    let batch_collector_task = tokio::spawn(async move {
        let (batch_client, connection) = tokio_postgres::connect(&db_uri, NoTls).await?;
        tokio::spawn(async move { connection.await });
        loop {
            let mut job_map = job_batching_hashmap.lock().await;
            let jobs: Vec<(i32, Vec<String>)> = job_map.drain().collect();
            // release lock
            drop(job_map);

            for (job_id, row_ids) in jobs {
                let job_result = batch_client
                    .query_one(
                        &format!("{job_query_sql} WHERE id=$1 AND canceled_at IS NULL"),
                        &[&job_id],
                    )
                    .await;
                if let Err(e) = job_result {
                    logger.error(&format!("Error while getting job {job_id}: {}", e));
                    continue;
                }
                let row = job_result.unwrap();
                let job = EmbeddingJob::new(row, data_path, &db_uri);

                if let Err(e) = &job {
                    logger.error(&format!("Error while creating job {job_id}: {e}"));
                    continue;
                }
                let mut job = job.unwrap();

                // update label in cache
                if let Some(label) = &job.label {
                    job_labels_map_r1
                        .write()
                        .await
                        .insert(job.id, label.clone());
                } else {
                    job_labels_map_r1.write().await.remove(&job.id);
                }

                job.set_is_init(false);
                let rows_len = row_ids.len();
                job.set_id_filter(&row_ids);
                job.set_row_ids(row_ids);
                logger.debug(&format!(
                    "Sending batch job {job_id} (len: {rows_len}) to embedding_worker"
                ));

                // Send in new tokio task to avoid blocking loop
                let logger = logger.clone();
                let job_tx = job_tx.clone();
                tokio::spawn(async move {
                    if let Err(e) = job_tx.send(job).await {
                        logger.error(&format!("Failed to send batch job: {job_id}: {e}"));
                    }
                });
            }

            // collect jobs every 10 seconds
            tokio::time::sleep(Duration::from_secs(10)).await;
        }
        #[allow(unreachable_code)]
        Ok::<(), anyhow::Error>(())
    });

    let (r1, _) = tokio::try_join!(insert_processor_task, batch_collector_task)?;
    r1?;

    Ok(())
}

async fn job_update_processor(
    db_uri: String,
    mut update_queue_rx: UnboundedReceiver<JobUpdateNotification>,
    job_insert_queue_tx: UnboundedSender<JobInsertNotification>,
    schema: String,
    table: String,
    jobs_map: Arc<JobEventHandlersMap>,
    job_batching_hashmap: Arc<JobBatchingHashMap>,
    client_jobs_map: Arc<ClientJobsMap>,
    logger: Arc<Logger>,
) -> AnyhowVoidResult {
    tokio::spawn(async move {
        while let Some(notification) = update_queue_rx.recv().await {
            let (client, connection) = tokio_postgres::connect(&db_uri, NoTls).await?;
            tokio::spawn(async move { connection.await });
            let full_table_name =  get_full_table_name(&schema, &table);
            let id = notification.id;
            // TODO:: Select pk here
            let row = client.query_one(&format!("SELECT dst_column, src_column as \"column\", \"table\", \"schema\", canceled_at, init_finished_at FROM {0} WHERE id=$1", &full_table_name), &[&id]).await?;
            let src_column = row.get::<&str, String>("column").to_owned();
            let out_column = row.get::<&str, String>("dst_column").to_owned();

            let canceled_at: Option<SystemTime> = row.get("canceled_at");
            let init_finished_at: Option<SystemTime> = row.get("init_finished_at");

            if !notification.generate_missing {
              logger.debug(&format!("Update job {id}: is_canceled: {}", canceled_at.is_some()));
            }

            if init_finished_at.is_some() {
              toggle_client_job(client_jobs_map.clone(), id, db_uri.clone(), "id".to_owned(), row.get::<&str, String>("column").to_owned(), row.get::<&str, String>("dst_column").to_owned(), row.get::<&str, String>("table").to_owned(), row.get::<&str, String>("schema").to_owned(), logger.level.clone(), &logger.label, Some(job_insert_queue_tx.clone()), canceled_at.is_none()).await?;
            } else if canceled_at.is_some() {
                // Cancel ongoing job
                let jobs = jobs_map.read().await;
                let job = jobs.get(&id);

                if let Some(tx) = job {
                   tx.send(JobEvent::Errored(JOB_CANCELLED_MESSAGE.to_string())).await?;
                }
                drop(jobs);

                // Cancel collected jobs
                let mut job_map = job_batching_hashmap.lock().await;
                job_map.remove(&id);
                drop(job_map);
            }

            if canceled_at.is_none() && notification.generate_missing {
                // this will be on startup to generate embeddings for rows that might be inserted
                // while daemon is offline
                job_insert_queue_tx.send(JobInsertNotification { id, generate_missing: true, filter: Some(get_missing_rows_filter(&src_column, &out_column)), limit: None, row_id: None })?;
            }
        }
        Ok(()) as AnyhowVoidResult
    })
    .await??;
    Ok(())
}

async fn create_data_path(logger: Arc<Logger>) -> &'static str {
    let tmp_path = "/tmp/lantern-daemon";
    let data_path = if cfg!(target_os = "macos") {
        "/usr/local/var/lantern-daemon"
    } else {
        "/var/lib/lantern-daemon"
    };

    let data_path_obj = Path::new(data_path);
    if data_path_obj.exists() {
        return data_path;
    }

    if fs::create_dir(data_path).await.is_ok() {
        return data_path;
    }

    logger.warn(&format!(
        "No write permission in directory {data_path}. Writing data to temp directory"
    ));
    let tmp_path_obj = Path::new(tmp_path);

    if tmp_path_obj.exists() {
        return tmp_path;
    }

    fs::create_dir(tmp_path).await.unwrap();
    tmp_path
}

async fn stop_all_client_jobs(
    client: Arc<Client>,
    logger: Arc<Logger>,
    client_jobs_map: Arc<ClientJobsMap>,
    db_uri: String,
    schema: String,
    jobs_table_name: String,
) -> AnyhowVoidResult {
    let rows = client
        .query(
            &format!("SELECT id, runtime, \"table\" FROM {jobs_table_name} WHERE init_finished_at IS NOT NULL AND canceled_at IS NULL ORDER BY id"),
            &[],
        )
        .await?;

    for row in rows {
        let job_id = row.get::<&str, i32>("id").to_owned();
        let job_runtime = row.get::<&str, &str>("runtime").to_owned();
        let job_table = row.get::<&str, &str>("table").to_owned();

        let task_logger = Arc::new(Logger::new(
            &format!("{}|{}|{:?}", logger.label, job_id, job_runtime),
            logger.level.clone(),
        ));

        if let Err(e) = stop_client_job(
            task_logger,
            client_jobs_map.clone(),
            &db_uri,
            job_id,
            &job_table,
            &schema,
            false,
        )
        .await
        {
            logger.error(&format!("Error while stopping job {job_id}: {e}"));
        };
    }
    Ok(())
}

pub async fn start(
    args: JobRunArgs,
    logger: Arc<Logger>,
    cancel_token: CancellationToken,
) -> AnyhowVoidResult {
    logger.info("Starting Embedding Jobs");

    let (mut main_db_client, connection) = tokio_postgres::connect(&args.uri, NoTls).await?;

    let connection_task = tokio::spawn(async move { connection.await });

    let notification_channel = "lantern_cloud_embedding_jobs_v2";
    let data_path = create_data_path(logger.clone()).await;

    let (insert_notification_queue_tx, insert_notification_queue_rx): (
        UnboundedSender<JobInsertNotification>,
        UnboundedReceiver<JobInsertNotification>,
    ) = mpsc::unbounded_channel();
    let (update_notification_queue_tx, update_notification_queue_rx): (
        UnboundedSender<JobUpdateNotification>,
        UnboundedReceiver<JobUpdateNotification>,
    ) = mpsc::unbounded_channel();
    let (job_queue_tx, job_queue_rx): (Sender<EmbeddingJob>, Receiver<EmbeddingJob>) =
        mpsc::channel(1);
    let table = args.table_name;

    //TODO:: Remove migration on next release
    let migration_sql = format!(
        "ALTER TABLE {full_table_name} ADD COLUMN IF NOT EXISTS label TEXT NULL;",
        full_table_name = get_full_table_name(&args.schema, &table)
    );
    // ======================================

    startup_hook(
        &mut main_db_client,
        &table,
        JOB_TABLE_DEFINITION,
        &args.schema,
        Some(EMB_LOCK_TABLE_NAME),
        None,
        None,
        Some(EMB_USAGE_TABLE_NAME),
        Some(USAGE_TABLE_DEFINITION),
        Some(&migration_sql),
        &notification_channel,
        logger.clone(),
    )
    .await?;

    connection_task.abort();
    let (main_db_client, connection) = tokio_postgres::connect(&args.uri, NoTls).await?;
    let main_db_client = Arc::new(main_db_client);

    let logger_clone = logger.clone();
    let jobs_table_name = get_full_table_name(&args.schema, &table);
    let main_db_uri = args.uri.clone();
    let schema = args.schema.clone();

    let jobs_map: Arc<JobEventHandlersMap> = Arc::new(RwLock::new(HashMap::new()));
    let client_jobs_map: Arc<ClientJobsMap> = Arc::new(RwLock::new(HashMap::new()));
    let job_labels_map: Arc<JobLabelsMap> = Arc::new(RwLock::new(HashMap::new()));

    let job_batching_hashmap: Arc<JobBatchingHashMap> = Arc::new(Mutex::new(HashMap::new()));

    tokio::try_join!(
        anyhow_wrap_connection::<NoTls>(connection),
        db_notification_listener(
            main_db_uri.clone(),
            &notification_channel,
            insert_notification_queue_tx.clone(),
            Some(update_notification_queue_tx.clone()),
            cancel_token.clone(),
            logger.clone(),
        ),
        job_insert_processor(
            insert_notification_queue_rx,
            insert_notification_queue_tx.clone(),
            job_queue_tx.clone(),
            main_db_uri.clone(),
            schema.clone(),
            table.clone(),
            args.label.clone().unwrap_or(String::new()),
            data_path,
            jobs_map.clone(),
            job_batching_hashmap.clone(),
            client_jobs_map.clone(),
            job_labels_map,
            logger.clone(),
        ),
        job_update_processor(
            main_db_uri.clone(),
            update_notification_queue_rx,
            insert_notification_queue_tx.clone(),
            schema.clone(),
            table.clone(),
            jobs_map.clone(),
            job_batching_hashmap.clone(),
            client_jobs_map.clone(),
            logger.clone(),
        ),
        embedding_worker(
            job_queue_rx,
            job_queue_tx.clone(),
            main_db_uri.clone(),
            schema.clone(),
            table.clone(),
            jobs_map.clone(),
            logger.clone(),
        ),
        collect_pending_jobs(
            main_db_client.clone(),
            update_notification_queue_tx.clone(),
            jobs_table_name.clone(),
        ),
        cancellation_handler(
            cancel_token.clone(),
            Some(move || async {
                stop_all_client_jobs(
                    main_db_client,
                    logger_clone,
                    client_jobs_map,
                    main_db_uri,
                    schema,
                    jobs_table_name,
                )
                .await?;

                Ok::<(), anyhow::Error>(())
            })
        )
    )?;

    Ok(())
}

use std::time::{Duration, Instant};

use lantern_cli::{
    daemon::{cli::DaemonArgs, start},
    logger::{LogLevel, Logger},
    types::AnyhowVoidResult,
    utils::quote_ident,
};
use pgrx::prelude::*;
use tokio::runtime::Runtime;
use tokio_util::sync::CancellationToken;

pub fn start_daemon(
    embeddings: bool,
    indexing: bool,
    autotune: bool,
) -> Result<bool, anyhow::Error> {
    let cancellation_token = CancellationToken::new();
    let (db, user, socket_path, port) = Spi::connect(|client| {
        let row = client
            .select(
                "
           SELECT current_database()::text AS db,
           (SELECT setting::text FROM pg_settings WHERE name = 'unix_socket_directories') AS socket_path,
           (SELECT setting::text FROM pg_settings WHERE name = 'port') AS port,
           (SELECT rolname::text FROM pg_roles WHERE rolsuper = true LIMIT 1) as user
           ",
                None,
                None,
            )?
            .first();

        let db = row.get_by_name::<String, &str>("db")?.unwrap();
        let socket_path = row.get_by_name::<String, &str>("socket_path")?.unwrap();
        let port = row.get_by_name::<String, &str>("port")?.unwrap();
        let user = row.get_by_name::<String, &str>("user")?.unwrap();

        Ok::<(String, String, String, String), anyhow::Error>((db, user, socket_path, port))
    })?;

    let connection_string = format!(
        "postgresql://{user}@{socket_path}:{port}/{db}",
        socket_path = socket_path.replace("/", "%2F")
    );

    std::thread::spawn(move || {
        let mut last_retry = Instant::now();
        let mut retry_interval = 5;
        loop {
            let logger = Logger::new("Lantern Daemon", LogLevel::Debug);
            let rt = Runtime::new().unwrap();
            let res = rt.block_on(start(
                DaemonArgs {
                    embeddings,
                    external_index: indexing,
                    autotune,
                    log_level: lantern_cli::daemon::cli::LogLevel::Debug,
                    databases_table: String::new(),
                    master_db: None,
                    master_db_schema: String::new(),
                    schema: String::from("_lantern_internal"),
                    target_db: Some(vec![connection_string.clone()]),
                },
                Some(logger.clone()),
                cancellation_token.clone(),
            ));

            if let Err(e) = res {
                eprintln!("{e}");
                logger.error(&format!("{e}"));
            }
            if last_retry.elapsed().as_secs() > retry_interval * 2 {
                // reset retry exponential backoff time if job was not failing constantly
                retry_interval = 10;
            }
            std::thread::sleep(Duration::from_secs(retry_interval));
            retry_interval *= 2;
            last_retry = Instant::now();
        }
    });

    Ok(true)
}

#[pg_extern(immutable, parallel_unsafe)]
fn add_embedding_job<'a>(
    table: &'a str,
    src_column: &'a str,
    dst_column: &'a str,
    embedding_model: &'a str,
    runtime: default!(&'a str, "'ort'"),
    runtime_params: default!(&'a str, "'{}'"),
    pk: default!(&'a str, "'id'"),
    schema: default!(&'a str, "'public'"),
) -> Result<i32, anyhow::Error> {
    let id: Option<i32> = Spi::get_one_with_args(
        &format!(
            r#"
          ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {dst_column} REAL[];
          INSERT INTO _lantern_internal.embedding_generation_jobs ("table", "schema", pk, src_column, dst_column, embedding_model, runtime, runtime_params) VALUES
          ($1, $2, $3, $4, $5, $6, $7, $8::jsonb) RETURNING id;
        "#,
            table = quote_ident(table),
            dst_column = quote_ident(dst_column)
        ),
        vec![
            (PgBuiltInOids::TEXTOID.oid(), table.into_datum()),
            (PgBuiltInOids::TEXTOID.oid(), schema.into_datum()),
            (PgBuiltInOids::TEXTOID.oid(), pk.into_datum()),
            (PgBuiltInOids::TEXTOID.oid(), src_column.into_datum()),
            (PgBuiltInOids::TEXTOID.oid(), dst_column.into_datum()),
            (PgBuiltInOids::TEXTOID.oid(), embedding_model.into_datum()),
            (PgBuiltInOids::TEXTOID.oid(), runtime.into_datum()),
            (PgBuiltInOids::TEXTOID.oid(), runtime_params.into_datum()),
        ],
    )?;

    Ok(id.unwrap())
}

#[pg_extern(immutable, parallel_safe)]
fn get_embedding_job_status<'a>(
    job_id: i32,
) -> Result<
    TableIterator<
        'static,
        (
            name!(status, Option<String>),
            name!(progress, Option<i16>),
            name!(error, Option<String>),
        ),
    >,
    anyhow::Error,
> {
    let tuple = Spi::get_three_with_args(
        r#"
          SELECT 
          CASE 
            WHEN init_failed_at IS NOT NULL THEN 'failed'
            WHEN canceled_at IS NOT NULL THEN 'canceled'
            WHEN init_finished_at IS NOT NULL THEN 'enabled'
            WHEN init_started_at IS NOT NULL THEN 'in_progress'
            ELSE 'queued'
          END AS status,
          init_progress as progress,
          init_failure_reason as error
          FROM _lantern_internal.embedding_generation_jobs
          WHERE id=$1;
        "#,
        vec![(PgBuiltInOids::INT4OID.oid(), job_id.into_datum())],
    )?;

    Ok(TableIterator::once(tuple))
}

#[pg_extern(immutable, parallel_safe)]
fn cancel_embedding_job<'a>(job_id: i32) -> AnyhowVoidResult {
    Spi::run_with_args(
        r#"
          UPDATE _lantern_internal.embedding_generation_jobs
          SET canceled_at=NOW()
          WHERE id=$1;
        "#,
        Some(vec![(PgBuiltInOids::INT4OID.oid(), job_id.into_datum())]),
    )?;

    Ok(())
}

#[pg_extern(immutable, parallel_safe)]
fn resume_embedding_job<'a>(job_id: i32) -> AnyhowVoidResult {
    Spi::run_with_args(
        r#"
          UPDATE _lantern_internal.embedding_generation_jobs
          SET canceled_at=NULL
          WHERE id=$1;
        "#,
        Some(vec![(PgBuiltInOids::INT4OID.oid(), job_id.into_datum())]),
    )?;

    Ok(())
}

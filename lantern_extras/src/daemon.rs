use std::time::{Duration, Instant};

use lantern_cli::{
    daemon::{cli::DaemonArgs, start},
    logger::{LogLevel, Logger},
    types::AnyhowVoidResult,
    utils::{get_full_table_name, quote_ident},
};
use pgrx::prelude::*;
use tokio::runtime::Runtime;
use tokio_util::sync::CancellationToken;

use crate::DAEMON_DATABASES;

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

    let mut target_dbs = vec![];

    if let Some(db_list) = DAEMON_DATABASES.get() {
        for db_name in db_list.to_str()?.split(",") {
            let connection_string = format!(
                "postgresql://{user}@{socket_path}:{port}/{db}",
                socket_path = socket_path.replace("/", "%2F"),
                db = db_name.trim()
            );
            target_dbs.push(connection_string);
        }
    } else {
        let connection_string = format!(
            "postgresql://{user}@{socket_path}:{port}/{db}",
            socket_path = socket_path.replace("/", "%2F")
        );
        target_dbs.push(connection_string);
    }

    std::thread::spawn(move || {
        let mut last_retry = Instant::now();
        let mut retry_interval = 5;
        loop {
            let logger = Logger::new("Lantern Daemon", LogLevel::Debug);
            let rt = Runtime::new().unwrap();
            let res = rt.block_on(start(
                DaemonArgs {
                    label: None,
                    embeddings,
                    external_index: indexing,
                    autotune,
                    log_level: lantern_cli::daemon::cli::LogLevel::Debug,
                    databases_table: String::new(),
                    master_db: None,
                    master_db_schema: String::new(),
                    schema: String::from("_lantern_internal"),
                    target_db: Some(target_dbs.clone()),
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
            table = get_full_table_name(schema, table),
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
    );

    if tuple.is_err() {
        return Ok(TableIterator::once((None, None, None)));
    }

    Ok(TableIterator::once(tuple.unwrap()))
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

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
pub mod tests {
    use crate::*;
    use std::time::Duration;

    #[pg_test]
    fn test_add_daemon_job() {
        Spi::connect(|mut client| {
            // wait for daemon
            std::thread::sleep(Duration::from_secs(1));
            client.update(
                "
                CREATE TABLE t1 (id serial primary key, title text);
                ",
                None,
                None,
            )?;
            let id = client.select("SELECT add_embedding_job('t1', 'title', 'title_embedding', 'BAAI/bge-small-en', 'ort', '{}', 'id', 'public')", None, None)?;

            let id: i32 = id.first().get(1)?.unwrap();

            assert_eq!(id, 1);
            Ok::<(), anyhow::Error>(())
        })
        .unwrap();
    }

    #[pg_test]
    fn test_get_daemon_job() {
        Spi::connect(|mut client| {
            // wait for daemon
            std::thread::sleep(Duration::from_secs(1));
            client.update(
                "
                CREATE TABLE t1 (id serial primary key, title text);
                ",
                None,
                None,
            )?;

            let id = client.update("SELECT add_embedding_job('t1', 'title', 'title_embedding', 'BAAI/bge-small-en', 'ort', '{}', 'id', 'public')", None, None)?;
            let id: i32 = id.first().get(1)?.unwrap();

            // queued
            let rows = client.select("SELECT status, progress, error FROM get_embedding_job_status($1)", None, Some(vec![(PgBuiltInOids::INT4OID.oid(), id.into_datum())]))?;
            let job = rows.first();

            let status: &str = job.get(1)?.unwrap();
            let progress: i16 = job.get(2)?.unwrap();
            let error: Option<&str> = job.get(3)?;

            assert_eq!(status, "queued");
            assert_eq!(progress, 0);
            assert_eq!(error, None);

            // Failed

            client.update("UPDATE _lantern_internal.embedding_generation_jobs SET init_failed_at=NOW(), init_failure_reason='test';", None, None)?;
            let rows = client.select("SELECT status, progress, error FROM get_embedding_job_status($1)", None, Some(vec![(PgBuiltInOids::INT4OID.oid(), id.into_datum())]))?;
            let job = rows.first();

            let status: &str = job.get(1)?.unwrap();
            let progress: i16 = job.get(2)?.unwrap();
            let error: &str = job.get(3)?.unwrap();

            assert_eq!(status, "failed");
            assert_eq!(progress, 0);
            assert_eq!(error, "test");

            // In progress
            client.update("UPDATE _lantern_internal.embedding_generation_jobs SET init_failed_at=NULL, init_failure_reason=NULL, init_progress=60, init_started_at=NOW();", None, None)?;
            let rows = client.select("SELECT status, progress, error FROM get_embedding_job_status($1)", None, Some(vec![(PgBuiltInOids::INT4OID.oid(), id.into_datum())]))?;
            let job = rows.first();

            let status: &str = job.get(1)?.unwrap();
            let progress: i16 = job.get(2)?.unwrap();
            let error: Option<&str> = job.get(3)?;

            assert_eq!(status, "in_progress");
            assert_eq!(progress, 60);
            assert_eq!(error, None);

            // Canceled
            client.update("UPDATE _lantern_internal.embedding_generation_jobs SET init_failed_at=NULL, init_failure_reason=NULL, init_progress=0, init_started_at=NULL, canceled_at=NOW();", None, None)?;
            let rows = client.select("SELECT status, progress, error FROM get_embedding_job_status($1)", None, Some(vec![(PgBuiltInOids::INT4OID.oid(), id.into_datum())]))?;
            let job = rows.first();

            let status: &str = job.get(1)?.unwrap();
            let progress: i16 = job.get(2)?.unwrap();
            let error: Option<&str> = job.get(3)?;

            assert_eq!(status, "canceled");
            assert_eq!(progress, 0);
            assert_eq!(error, None);

            // Enabled
            client.update("UPDATE _lantern_internal.embedding_generation_jobs SET init_failed_at=NULL, init_failure_reason=NULL, init_progress=100, init_started_at=NULL, canceled_at=NULL, init_finished_at=NOW();", None, None)?;
            let rows = client.select("SELECT status, progress, error FROM get_embedding_job_status($1)", None, Some(vec![(PgBuiltInOids::INT4OID.oid(), id.into_datum())]))?;
            let job = rows.first();

            let status: &str = job.get(1)?.unwrap();
            let progress: i16 = job.get(2)?.unwrap();
            let error: Option<&str> = job.get(3)?;

            assert_eq!(status, "enabled");
            assert_eq!(progress, 100);
            assert_eq!(error, None);

            Ok::<(), anyhow::Error>(())
        })
        .unwrap();
    }

    #[pg_test]
    fn test_cancel_daemon_job() {
        Spi::connect(|mut client| {
            // wait for daemon
            std::thread::sleep(Duration::from_secs(1));
            client.update(
                "
                CREATE TABLE t1 (id serial primary key, title text);
                ",
                None,
                None,
            )?;
            let id = client.update("SELECT add_embedding_job('t1', 'title', 'title_embedding', 'BAAI/bge-small-en', 'ort', '{}', 'id', 'public')", None, None)?;
            let id: i32 = id.first().get(1)?.unwrap();
            client.update("SELECT cancel_embedding_job($1)", None, Some(vec![(PgBuiltInOids::INT4OID.oid(), id.into_datum())]))?;
            let rows = client.select("SELECT status, progress, error FROM get_embedding_job_status($1)", None, Some(vec![(PgBuiltInOids::INT4OID.oid(), id.into_datum())]))?;
            let job = rows.first();

            let status: &str = job.get(1)?.unwrap();
            let progress: i16 = job.get(2)?.unwrap();
            let error: Option<&str> = job.get(3)?;

            assert_eq!(status, "canceled");
            assert_eq!(progress, 0);
            assert_eq!(error, None);
            Ok::<(), anyhow::Error>(())
        })
        .unwrap();
    }

    #[pg_test]
    fn test_resume_daemon_job() {
        Spi::connect(|mut client| {
            // wait for daemon
            std::thread::sleep(Duration::from_secs(1));
            client.update(
                "
                CREATE TABLE t1 (id serial primary key, title text);
                ",
                None,
                None,
            )?;
            let id = client.update("SELECT add_embedding_job('t1', 'title', 'title_embedding', 'BAAI/bge-small-en', 'ort', '{}', 'id', 'public')", None, None)?;
            let id: i32 = id.first().get(1)?.unwrap();
            client.update("SELECT cancel_embedding_job($1)", None, Some(vec![(PgBuiltInOids::INT4OID.oid(), id.into_datum())]))?;
            client.update("SELECT resume_embedding_job($1)", None, Some(vec![(PgBuiltInOids::INT4OID.oid(), id.into_datum())]))?;
            let rows = client.select("SELECT status, progress, error FROM get_embedding_job_status($1)", None, Some(vec![(PgBuiltInOids::INT4OID.oid(), id.into_datum())]))?;
            let job = rows.first();

            let status: &str = job.get(1)?.unwrap();
            let progress: i16 = job.get(2)?.unwrap();
            let error: Option<&str> = job.get(3)?;

            assert_eq!(status, "queued");
            assert_eq!(progress, 0);
            assert_eq!(error, None);
            Ok::<(), anyhow::Error>(())
        })
        .unwrap();
    }
}

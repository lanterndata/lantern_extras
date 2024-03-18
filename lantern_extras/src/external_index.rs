use lantern_cli::external_index;
use lantern_cli::external_index::cli::{CreateIndexArgs, UMetricKind};
use pgrx::prelude::*;
use rand::Rng;

fn validate_index_param(param_name: &str, param_val: i32, min: i32, max: i32) {
    if param_val < min || param_val > max {
        error!("{param_name} should be in range [{min}, {max}]");
    }
}

unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::core::slice::from_raw_parts((p as *const T) as *const u8, ::core::mem::size_of::<T>())
}

#[pg_extern(immutable, parallel_unsafe)]
fn lantern_create_external_index<'a>(
    column: &'a str,
    table: &'a str,
    schema: default!(&'a str, "'public'"),
    metric_kind: default!(&'a str, "'l2sq'"),
    dim: default!(i32, 0),
    m: default!(i32, 16),
    ef_construction: default!(i32, 16),
    ef: default!(i32, 16),
    pq: default!(bool, false),
    index_name: default!(&'a str, "''"),
) -> Result<(), anyhow::Error> {
    validate_index_param("ef", ef, 1, 400);
    validate_index_param("ef_construction", ef_construction, 1, 400);
    validate_index_param("ef_construction", ef_construction, 1, 400);
    validate_index_param("m", m, 2, 128);

    if dim != 0 {
        validate_index_param("dim", dim, 1, 2000);
    }

    let (db, user, socket_path, port, data_dir) = Spi::connect(|client| {
        let row = client
            .select(
                "
           SELECT current_database()::text AS db,
           current_user::text AS user,
           (SELECT setting::text FROM pg_settings WHERE name = 'unix_socket_directories') AS socket_path,
           (SELECT setting::text FROM pg_settings WHERE name = 'port') AS port,
           (SELECT setting::text FROM pg_settings WHERE name = 'data_directory') as data_dir",
                None,
                None,
            )?
            .first();

        let db = row.get_by_name::<String, &str>("db")?.unwrap();
        let user = row.get_by_name::<String, &str>("user")?.unwrap();
        let socket_path = row.get_by_name::<String, &str>("socket_path")?.unwrap();
        let port = row.get_by_name::<String, &str>("port")?.unwrap();
        let data_dir = row.get_by_name::<String, &str>("data_dir")?.unwrap();

        Ok::<(String, String, String, String, String), anyhow::Error>((
            db,
            user,
            socket_path,
            port,
            data_dir,
        ))
    })?;

    let connection_string = format!("dbname={db} host={socket_path} user={user} port={port}");

    let index_name = if index_name == "" {
        None
    } else {
        Some(index_name.to_owned())
    };

    let mut rng = rand::thread_rng();
    let index_path = format!("{data_dir}/ldb-index-{}.usearch", rng.gen_range(0..1000));

    let res = external_index::create_usearch_index(
        &CreateIndexArgs {
            import: true,
            out: index_path,
            table: table.to_owned(),
            schema: schema.to_owned(),
            metric_kind: UMetricKind::from(metric_kind)?,
            efc: ef_construction as usize,
            ef: ef as usize,
            m: m as usize,
            uri: connection_string,
            column: column.to_owned(),
            dims: dim as usize,
            index_name,
            remote_database: false,
            pq,
        },
        None,
        None,
        None,
    );

    if let Err(e) = res {
        error!("{e}");
    }

    Ok(())
}

#[pg_schema]
mod lantern_extras {
    use super::lantern_create_external_index;
    use lantern_cli::{
        external_index::{
            cli::UMetricKind, create_index_from_stream, utils, IndexOptions, ScalarKind,
        },
        logger::{LogLevel, Logger},
        utils::{get_full_table_name, quote_ident},
    };
    use pgrx::pg_sys::ItemPointerData;
    use pgrx::prelude::*;
    use pgrx::{PgBuiltInOids, PgRelation, Spi};
    use rand::Rng;
    use std::sync::Arc;
    use std::time::Instant;

    #[pg_extern(immutable, parallel_unsafe)]
    fn _reindex_external_index<'a>(
        index: PgRelation,
        metric_kind: &'a str,
        dim: i32,
        m: i32,
        ef_construction: i32,
        ef: i32,
        pq: default!(bool, false),
    ) -> Result<(), anyhow::Error> {
        let index_name = index.name().to_owned();
        let schema = index.namespace().to_owned();
        let (table, column) = Spi::connect(|client| {
            let rows = client.select(
                "
                SELECT idx.indrelid::regclass::text   AS table_name,
                       att.attname::text              AS column_name
                FROM   pg_index AS idx
                       JOIN pg_attribute AS att
                         ON att.attrelid = idx.indrelid
                            AND att.attnum = ANY(idx.indkey)
                WHERE  idx.indexrelid = $1",
                None,
                Some(vec![(
                    PgBuiltInOids::OIDOID.oid(),
                    index.oid().into_datum(),
                )]),
            )?;

            if rows.len() == 0 {
                error!("Index with oid {:?} not found", index.oid());
            }

            let row = rows.first();

            let table = row.get_by_name::<String, &str>("table_name")?.unwrap();
            let column = row.get_by_name::<String, &str>("column_name")?.unwrap();
            Ok::<(String, String), anyhow::Error>((table, column))
        })?;

        drop(index);
        lantern_create_external_index(
            &column,
            &table,
            &schema,
            metric_kind,
            dim,
            m,
            ef_construction,
            ef,
            pq,
            &index_name,
        )
    }

    #[pg_extern(immutable, parallel_unsafe)]
    fn _create_external_index<'a>(
        column: &'a str,
        table: &'a str,
        schema: default!(&'a str, "'public'"),
        metric_kind: default!(&'a str, "'l2sq'"),
        dim: default!(i32, 0),
        m: default!(i32, 16),
        ef_construction: default!(i32, 16),
        ef: default!(i32, 16),
        pq: default!(bool, false),
    ) -> Result<String, anyhow::Error> {
        // TODO:::::
        // Refactor code
        // Add pq part
        // Merge similart logic with cli index
        // Add interrupt listener to cancel indexing
        //
        let index_path = Spi::connect(|client| {
            let dim = dim as usize;
            let full_table_name = get_full_table_name(schema, table);
            let logger = Arc::new(Logger::new("Lantern Index", LogLevel::Debug));
            let rows = client.select(&format!("SELECT ARRAY_LENGTH({col}, 1) as dim FROM {full_table_name} WHERE {col} IS NOT NULL",col=quote_ident(column)), Some(1), None)?;

            if rows.len() == 0 {
                anyhow::bail!("Cannot create an external index on empty table");
            }

            let row = rows.first();

            let dimensions = row.get_by_name::<i32, &str>("dim")?.unwrap() as usize;

            if dim != 0 && dimensions != dim {
                anyhow::bail!("Infered dimensions ({dimensions}) does not match with the provided dimensions ({dim})");
            }

            let metric = UMetricKind::from(metric_kind)?.value();

            // TODO fetch pq_codebook if pq is true
            let mut pq_codebook: *const f32 = std::ptr::null();
            let options = IndexOptions {
                dimensions,
                metric,
                quantization: ScalarKind::F32,
                multi: false,
                connectivity: m as usize,
                expansion_add: ef_construction as usize,
                expansion_search: ef as usize,

                num_threads: 0, // automatic

                // note: pq_construction and pq_output distinction is not yet implemented in usearch
                // in the future, if pq_construction is false, we will use full vectors in memory (and
                // require large memory for construction) but will output pq-quantized graph
                //
                // currently, regardless of pq_construction value, as long as pq_output is true,
                // we construct a pq_quantized index using quantized values during construction
                pq_construction: pq,
                pq_output: pq,
                num_centroids: 0,
                num_subvectors: 0,
                codebook: pq_codebook,
            };
            let start_time = Instant::now();

            let rows = client.select(
                &format!(
                    "SELECT COUNT(*) as cnt FROM {full_table_name} WHERE {} IS NOT NULL;",
                    quote_ident(column)
                ),
                None,
                None,
            )?;
            let count: i64 = rows.first().get_by_name("cnt")?.unwrap();
            info!("Count estimation took {}s", start_time.elapsed().as_secs());

            let data_dir = client.select(
                "SELECT setting::text as data_dir FROM pg_settings WHERE name = 'data_directory'",
                None,
                None,
            )?;

            let data_dir: Option<String> = data_dir.first().get_by_name("data_dir")?;

            if data_dir.is_none() {
                anyhow::bail!("Could not get data_dir");
            }

            let data_dir = data_dir.unwrap();

            let mut rng = rand::thread_rng();
            let index_path = format!("{data_dir}/ldb-index-{}.usearch", rng.gen_range(0..1000));
            let (tx, progress_tx, index_arc, handles) = create_index_from_stream(
                logger.clone(),
                None,
                None,
                true,
                &options,
                count as usize,
            )?;

            // With portal we can execute a query and poll values from it in chunks
            let mut cursor = client.open_cursor(
                &format!(
                    "SELECT ctid, {col} as v FROM {table} WHERE {col} IS NOT NULL;",
                    col = quote_ident(column),
                    table = get_full_table_name(schema, table)
                ),
                None,
            );

            loop {
                // poll 2000 rows from portal and send it to worker threads via channel
                let rows = cursor.fetch(2000)?;

                if rows.len() == 0 {
                    break;
                }

                // TODO:: check for interrupt
                tx.send(
                    rows.map(|r| {
                        let label: u64 = unsafe {
                            utils::bytes_to_integer_le(super::any_as_u8_slice(
                                &r.get_by_name::<ItemPointerData, &str>("ctid")
                                    .unwrap()
                                    .unwrap(),
                            ))
                        };

                        (
                            label,
                            r.get_by_name::<Vec<f32>, &str>("v").unwrap().unwrap(),
                        )
                    })
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

            info!("Indexing took {}s", start_time.elapsed().as_secs());

            index_arc.save(&index_path)?;

            info!("Index saved under {index_path}");

            Ok::<String, anyhow::Error>(index_path)
        })?;

        Ok(index_path)
    }
}

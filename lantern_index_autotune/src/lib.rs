use lantern_logger::{LogLevel, Logger};
use lantern_utils::{get_full_table_name, quote_ident};
use postgres::{Client, NoTls, Row};

pub mod cli;

type AnyhowVoidResult = Result<(), anyhow::Error>;

static INTERNAL_SCHEMA_NAME: &'static str = "_lantern_internal";

struct IndexParams {
    ef: usize,
    ef_construction: usize,
    m: usize,
}

fn create_test_table(
    client: &mut Client,
    tmp_table_name: &str,
    src_table_name: &str,
) -> AnyhowVoidResult {
    client.batch_execute(&format!(
        "
      CREATE SCHEMA IF NOT EXISTS {INTERNAL_SCHEMA_NAME};
      DROP TABLE IF EXISTS {tmp_table_name};
      SELECT * INTO {tmp_table_name} FROM {src_table_name} LIMIT 100000;
    "
    ))?;
    Ok(())
}

fn calculate_ground_truth(
    client: &mut Client,
    pk: &str,
    emb_col: &str,
    tmp_table_name: &str,
    truth_table_name: &str,
    distance_function: &str,
    k: u16,
) -> AnyhowVoidResult {
    client.batch_execute(&format!(
        "
         DROP TABLE IF EXISTS {truth_table_name};
         SELECT tmp.{pk}, tmp.{emb_col}, ARRAY(SELECT {pk} FROM {tmp_table_name} tmp2 ORDER BY {distance_function}(tmp.{emb_col}, tmp2.{emb_col}) LIMIT {k}) as neighbors
         INTO {truth_table_name}
         FROM {tmp_table_name} tmp
         WHERE {pk} IN (SELECT {pk} FROM {tmp_table_name} ORDER BY RANDOM() LIMIT 100)",
        pk = quote_ident(pk),
        emb_col = quote_ident(emb_col),
    ))?;
    Ok(())
}

pub fn autotune_index(args: &cli::IndexAutotuneArgs, logger: Option<Logger>) -> AnyhowVoidResult {
    let logger = logger.unwrap_or(Logger::new("Lantern Index", LogLevel::Debug));
    let mut client = Client::connect(&args.uri, NoTls)?;
    let src_table_name = get_full_table_name(&args.schema, &args.table);
    let tmp_table_name =
        get_full_table_name(INTERNAL_SCHEMA_NAME, &format!("_test_{}", &args.table));
    let truth_table_name =
        get_full_table_name(INTERNAL_SCHEMA_NAME, &format!("_truth_{}", &args.table));
    // select from src table to target table limit 100k
    create_test_table(&mut client, &tmp_table_name, &src_table_name)?;
    calculate_ground_truth(
        &mut client,
        &args.pk,
        &args.column,
        &tmp_table_name,
        &truth_table_name,
        &args.metric_kind.sql_function(),
        args.k,
    )?;

    let index_variants = vec![
        IndexParams {
            ef: 64,
            ef_construction: 32,
            m: 16,
        }, // light
        IndexParams {
            ef: 64,
            ef_construction: 64,
            m: 32,
        }, // medium
        IndexParams {
            ef: 128,
            ef_construction: 128,
            m: 48,
        }, // correct
    ];

    // for variant in &index_variants {
    //     lantern_create_index::
    //     // create index
    //     // calculate recall and latency using ground truth
    //     // export results to result table
    //
    // }

    // if create index is passed as true create index with the best result
    Ok(())
}

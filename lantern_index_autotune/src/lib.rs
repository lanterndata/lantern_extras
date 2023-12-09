use std::{collections::HashSet, time::Instant};
use uuid::Uuid;

use lantern_create_index::cli::CreateIndexArgs;
use lantern_logger::{LogLevel, Logger};
use lantern_utils::{append_params_to_uri, get_full_table_name, quote_ident};
use postgres::{types::ToSql, Client, NoTls};
use rand::Rng;

pub mod cli;

type AnyhowVoidResult = Result<(), anyhow::Error>;
type GroundTruth = Vec<(Vec<f32>, Vec<String>)>;
pub type ProgressCbFn = Box<dyn Fn(u8) + Send + Sync>;

static INTERNAL_SCHEMA_NAME: &'static str = "_lantern_internal";
static CONNECTION_PARAMS: &'static str = "connect_timeout=10";

#[derive(Debug)]
struct IndexParams {
    ef: usize,
    ef_construction: usize,
    m: usize,
}

#[derive(Debug, Clone)]
struct AutotuneResult {
    job_id: String,
    model_name: Option<String>,
    metric_kind: String,
    ef: i32,
    ef_construction: i32,
    m: i32,
    k: i32,
    dim: i32,
    sample_size: i32,
    recall: f64,
    latency: i32,
    indexing_duration: i32,
}

fn create_test_table(
    client: &mut Client,
    tmp_table_name: &str,
    src_table_name: &str,
    column_name: &str,
    test_data_size: usize,
) -> Result<usize, anyhow::Error> {
    client.batch_execute(&format!(
        "
      CREATE SCHEMA IF NOT EXISTS {INTERNAL_SCHEMA_NAME};
      DROP TABLE IF EXISTS {tmp_table_name};
      SELECT * INTO {tmp_table_name} FROM {src_table_name} LIMIT {test_data_size};
    "
    ))?;
    let dims = client.query_one(
        &format!(
            "SELECT ARRAY_LENGTH({column_name}, 1) FROM {tmp_table_name} LIMIT 1",
            column_name = quote_ident(column_name)
        ),
        &[],
    )?;
    let dims: i32 = dims.get(0);

    if dims == 0 {
        anyhow::bail!("Column does not have dimensions");
    }

    Ok(dims as usize)
}

fn create_results_table(client: &mut Client, result_table_full_name: &str) -> AnyhowVoidResult {
    client.execute(&format!("CREATE TABLE IF NOT EXISTS {result_table_full_name} (id SERIAL PRIMARY KEY, job_id TEXT, model_name TEXT, ef INT, ef_construction INT, m INT, k INT, recall FLOAT, latency INT, dim INT, sample_size INT, indexing_duration INT, metric_kind TEXT)"), &[])?;
    Ok(())
}

fn export_results(
    client: &mut Client,
    result_table_full_name: &str,
    autotune_results: Vec<AutotuneResult>,
) -> AnyhowVoidResult {
    let mut query = format!("INSERT INTO {result_table_full_name} (job_id, model_name, ef, ef_construction, m, k, recall, latency, dim, sample_size, indexing_duration, metric_kind) VALUES ");
    let mut param_idx = 1;
    let params: Vec<&(dyn ToSql + Sync)> = autotune_results
        .iter()
        .flat_map(|row| {
            let comma_str = if param_idx == 1 { "" } else { "," };
            query = format!(
                "{}{} (${},${},${},${},${},${},${},${},${},${},${},${})",
                query,
                comma_str,
                param_idx,
                param_idx + 1,
                param_idx + 2,
                param_idx + 3,
                param_idx + 4,
                param_idx + 5,
                param_idx + 6,
                param_idx + 7,
                param_idx + 8,
                param_idx + 9,
                param_idx + 10,
                param_idx + 11,
            );

            param_idx += 12;
            [
                &row.job_id as &(dyn ToSql + Sync),
                &row.model_name as &(dyn ToSql + Sync),
                &row.ef,
                &row.ef_construction,
                &row.m,
                &row.k,
                &row.recall,
                &row.latency,
                &row.dim,
                &row.sample_size,
                &row.indexing_duration,
                &row.metric_kind,
            ]
        })
        .collect();

    client.execute(&query, &params[..])?;

    Ok(())
}

fn get_existing_results_for_model(
    client: &mut Client,
    model_name: &str,
    k: i32,
    sample_size: i32,
    result_table_name: &str,
    result_table_schema: &str,
    result_table_full_name: &str,
) -> Result<Option<Vec<AutotuneResult>>, anyhow::Error> {
    let table_exists = client.query_one("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name=$1 AND table_schema=$2) AS table_existence", &[&result_table_name, &result_table_schema])?;

    if !table_exists.get::<usize, bool>(0) {
        return Ok(None);
    }

    let rows = client.query(
        &format!("SELECT * FROM {result_table_full_name} WHERE job_id=(SELECT job_id FROM {result_table_full_name} WHERE model_name=$1 AND k>=$2 AND sample_size>=$3 GROUP BY job_id, model_name LIMIT 1)"),
        &[&model_name, &k, &sample_size],
    )?;
    if rows.len() == 0 {
        return Ok(None);
    }

    let mut res: Vec<AutotuneResult> = Vec::with_capacity(rows.len());

    for row in rows {
        res.push(AutotuneResult {
            job_id: row.get::<&str, &str>("job_id").to_owned(),
            model_name: row
                .get::<&str, Option<&str>>("model_name")
                .map(str::to_string),
            metric_kind: row.get::<&str, &str>("metric_kind").to_owned(),
            ef: row.get::<&str, i32>("ef"),
            ef_construction: row.get::<&str, i32>("ef_construction"),
            m: row.get::<&str, i32>("m"),
            k: row.get::<&str, i32>("k"),
            dim: row.get::<&str, i32>("dim"),
            sample_size: row.get::<&str, i32>("sample_size"),
            recall: row.get::<&str, f64>("recall"),
            latency: row.get::<&str, i32>("latency"),
            indexing_duration: row.get::<&str, i32>("indexing_duration"),
        });
    }

    Ok(Some(res))
}

fn find_best_variant(autotune_results: &Vec<AutotuneResult>, target_recall: f64) -> AutotuneResult {
    let mut results_clone = autotune_results.clone();
    // Firstly we will sort the results by recall in descending order
    results_clone.sort_by(|a, b| b.recall.partial_cmp(&a.recall).unwrap());
    // Then we will filter the results which are matching the target recall
    let filtered_results: Vec<&AutotuneResult> = results_clone
        .iter()
        .filter(|el| el.recall >= target_recall)
        .collect();

    // If no match is found then we will return the result with highest recall
    if filtered_results.len() == 0 {
        return results_clone.first().unwrap().clone();
    }

    // Then we will sort by latency + index creation time
    // So if the target recall is met we can create an index which will be faster
    let mut filtered_results: Vec<(i32, &AutotuneResult)> = filtered_results
        .iter()
        .map(|r| (r.latency + r.indexing_duration, *r))
        .collect();

    filtered_results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    return filtered_results.first().unwrap().1.clone();
}

fn calculate_ground_truth(
    client: &mut Client,
    pk: &str,
    emb_col: &str,
    tmp_table_name: &str,
    truth_table_name: &str,
    distance_function: &str,
    k: u16,
) -> Result<GroundTruth, anyhow::Error> {
    client.batch_execute(&format!(
        "
         DROP TABLE IF EXISTS {truth_table_name};
         SELECT tmp.{pk} as id, tmp.{emb_col}::real[] as v, ARRAY(SELECT {pk}::text FROM {tmp_table_name} tmp2 ORDER BY {distance_function}(tmp.{emb_col}, tmp2.{emb_col}) LIMIT {k}) as neighbors
         INTO {truth_table_name}
         FROM {tmp_table_name} tmp
         WHERE {pk} IN (SELECT {pk} FROM {tmp_table_name} ORDER BY RANDOM() LIMIT 10)",
        pk = quote_ident(pk),
        emb_col = quote_ident(emb_col),
    ))?;
    let ground_truth = client.query(
        &format!(
            "SELECT {emb_col}, neighbors FROM {truth_table_name}",
            emb_col = quote_ident(emb_col)
        ),
        &[],
    )?;

    Ok(ground_truth
        .iter()
        .map(|row| {
            return (
                row.get::<usize, Vec<f32>>(0),
                row.get::<usize, Vec<String>>(1),
            );
        })
        .collect())
}

fn calculate_recall_and_latency(
    client: &mut Client,
    ground_truth: &GroundTruth,
    test_table_name: &str,
    k: u16,
) -> Result<(f32, usize), anyhow::Error> {
    let mut recall: f32 = 0.0;
    let mut latency: usize = 0;

    for (query, neighbors) in ground_truth {
        let start = Instant::now();
        let rows = client.query(
            &format!("SELECT id::text FROM {test_table_name} ORDER BY $1<->v LIMIT {k}"),
            &[query],
        )?;
        latency += start.elapsed().as_millis() as usize;

        let truth: HashSet<String> = neighbors.into_iter().map(|s| s.to_owned()).collect();
        let result: HashSet<String> = rows
            .into_iter()
            .map(|r| r.get::<usize, &str>(0).to_owned())
            .collect();
        let intersection = truth.intersection(&result).collect::<Vec<_>>();

        let query_recall = (intersection.len() as f32 / truth.len() as f32) * 100.0;
        recall += query_recall;
    }

    recall = recall / ground_truth.len() as f32;
    latency = latency / ground_truth.len();
    Ok((recall, latency))
}

fn print_results(logger: &Logger, results: &Vec<AutotuneResult>) {
    let job_id = &results[0].job_id;
    logger.info(&format!("{:=<10} Results for job {job_id} {:=<10}", "", ""));
    for result in results {
        logger.info(&format!(
            "result(recall={recall}%, latency={latency}ms, indexing_duration={indexing_duration}s) index_params(m={m}, ef={ef}, ef_construction={ef_construction})",
            recall = result.recall,
            latency = result.latency,
            indexing_duration = result.indexing_duration,
            ef = result.ef,
            ef_construction = result.ef_construction,
            m = result.m
        ));
    }
}

fn report_progress(progress_cb: &Option<ProgressCbFn>, logger: &Logger, progress: u8) {
    logger.debug(&format!("Progress {progress}%"));
    if progress_cb.is_some() {
        let cb = progress_cb.as_ref().unwrap();
        cb(progress);
    }
}

pub fn autotune_index(
    args: &cli::IndexAutotuneArgs,
    progress_cb: Option<ProgressCbFn>,
    logger: Option<Logger>,
) -> AnyhowVoidResult {
    let logger = logger.unwrap_or(Logger::new("Lantern Index", LogLevel::Debug));

    let uri = append_params_to_uri(&args.uri, CONNECTION_PARAMS);
    let mut client = Client::connect(&uri, NoTls)?;

    let mut progress: u8 = 0;
    let src_table_name = get_full_table_name(&args.schema, &args.table);
    let tmp_table_name = format!("_test_{}", &args.table);
    let tmp_table_full_name = get_full_table_name(INTERNAL_SCHEMA_NAME, &tmp_table_name);
    let truth_table_name =
        get_full_table_name(INTERNAL_SCHEMA_NAME, &format!("_truth_{}", &args.table));
    let result_table_name = &args.export_table_name;
    let result_table_full_name = get_full_table_name(&args.export_schema_name, &result_table_name);

    // Create table where we will create intermediate index results
    // This temp table will contain random subset of rows in size of test_data_size from source table
    let column_dims = create_test_table(
        &mut client,
        &tmp_table_full_name,
        &src_table_name,
        &args.column,
        args.test_data_size,
    )?;
    // Calculate ground truth for the data
    // Using sequential scan over temp table
    // It will have the following structure (id: INT, vector: REAL[], neighbors: INTEGER[])
    // This table will be used to calculate recall for index variant
    let ground_truth = calculate_ground_truth(
        &mut client,
        &args.pk,
        &args.column,
        &tmp_table_full_name,
        &truth_table_name,
        &args.metric_kind.sql_function(),
        args.k,
    )?;

    // These are the index variations we are going to create
    // We will sequentially iterate over this vector and create an index with each variant
    // Then calculate recall and latency for each one
    let index_variants = vec![
        IndexParams {
            ef: 64,
            ef_construction: 32,
            m: 6,
        },
        IndexParams {
            ef: 64,
            ef_construction: 40,
            m: 8,
        },
        IndexParams {
            ef: 64,
            ef_construction: 48,
            m: 12,
        },
        IndexParams {
            ef: 76,
            ef_construction: 60,
            m: 16,
        },
        IndexParams {
            ef: 96,
            ef_construction: 96,
            m: 32,
        },
        IndexParams {
            ef: 128,
            ef_construction: 128,
            m: 48,
        },
    ];

    // Create random index file name and job_id if not provided
    let mut rng = rand::thread_rng();
    let index_path = format!("/tmp/index-autotune-{}.usearch", rng.gen_range(0..1000));
    let index_name = format!("lantern_autotune_idx_{}", rng.gen_range(0..1000));
    let uuid = Uuid::new_v4().to_string();
    let job_id = args.job_id.as_ref().unwrap_or(&uuid);

    // Create db client for exporting and finding existing results
    let mut autotune_results: Vec<AutotuneResult> = Vec::with_capacity(index_variants.len());
    let export_uri = args.export_db_uri.clone().unwrap_or(args.uri.clone());
    let uri = append_params_to_uri(&export_uri, CONNECTION_PARAMS);
    let mut export_client = Client::connect(&uri, NoTls)?;

    // If the model name is provided we will check if we already have results for that model
    // And if so we will instead use precomputed results
    if let Some(model_name) = &args.model_name {
        let existing_results = get_existing_results_for_model(
            &mut export_client,
            &model_name,
            args.k as i32,
            args.test_data_size as i32,
            &result_table_name,
            &args.export_schema_name,
            &result_table_full_name,
        )?;

        if let Some(results) = existing_results {
            logger.info(&format!("Found existing results for model '{model_name}'"));
            autotune_results = results;
            for result in &mut autotune_results {
                // set new unique job id for the job
                result.job_id = job_id.clone();
            }
        }
    }

    progress += 5;
    report_progress(&progress_cb, &logger, progress);

    // 30% from progress is reserved for result export and index creation
    let progress_per_iter = (100 - progress - 30) / index_variants.len() as u8;
    if autotune_results.len() == 0 {
        // If no existing results were found, we will iterate over the variations and do the following:
        // 1. DROP previous iteration index if exists (if not the first iteration)
        // 2. Start external index creation with lantern_create_index.
        //    It will have import flag, which means it will import the index file using large
        //    objects
        // 3. Calculate the index creation time, latency and recall for this variation
        // 4. Save the result in results vector
        for variant in &index_variants {
            client.execute(
                &format!(
                    "DROP INDEX IF EXISTS {index_name}",
                    index_name = get_full_table_name(INTERNAL_SCHEMA_NAME, &index_name)
                ),
                &[],
            )?;
            let start = Instant::now();
            lantern_create_index::create_usearch_index(
                &CreateIndexArgs {
                    import: true,
                    out: index_path.clone(),
                    table: tmp_table_name.clone(),
                    schema: INTERNAL_SCHEMA_NAME.to_owned(),
                    metric_kind: args.metric_kind.clone(),
                    efc: variant.ef_construction,
                    ef: variant.ef,
                    m: variant.m,
                    uri: uri.clone(),
                    column: args.column.clone(),
                    dims: column_dims as usize,
                    index_name: Some(index_name.clone()),
                },
                Some(Logger::new(&logger.label, LogLevel::Info)),
                None,
            )?;
            let indexing_duration = start.elapsed().as_secs() as usize;
            let (recall, latency) = calculate_recall_and_latency(
                &mut client,
                &ground_truth,
                &tmp_table_full_name,
                args.k,
            )?;
            autotune_results.push(AutotuneResult {
                job_id: job_id.clone(),
                metric_kind: args.metric_kind.sql_function(),
                ef: variant.ef as i32,
                ef_construction: variant.ef_construction as i32,
                m: variant.m as i32,
                k: args.k as i32,
                dim: column_dims as i32,
                sample_size: args.test_data_size as i32,
                recall: recall as f64,
                latency: latency as i32,
                indexing_duration: indexing_duration as i32,
                model_name: args.model_name.clone(),
            });
            progress += progress_per_iter;
            report_progress(&progress_cb, &logger, progress);

            if recall >= 99.9 {
                break;
            }
        }
    }

    report_progress(&progress_cb, &logger, 70);

    // Print autotune results
    print_results(&logger, &autotune_results);

    // Drop the tables we have created
    client.batch_execute(&format!(
        "
        DROP TABLE IF EXISTS {tmp_table_full_name} CASCADE;
        DROP TABLE IF EXISTS {truth_table_name} CASCADE;
    "
    ))?;

    // if the export flag is provided
    // we will create the export table
    // and insert the results into that table
    if args.export {
        create_results_table(&mut export_client, &result_table_full_name)?;
        export_results(
            &mut export_client,
            &result_table_full_name,
            autotune_results.clone(),
        )?;
        logger.debug(&format!(
            "Results for job {job_id} exported to {result_table_name}"
        ));
    }

    // If create_index is passed
    // We will find the best variant for target recall
    // And create index with that variant
    if args.create_index {
        report_progress(&progress_cb, &logger, 80);
        let best_result = find_best_variant(&autotune_results, args.recall as f64);
        logger.debug(&format!(
            "Creating index with the best result for job {job_id}"
        ));
        let start = Instant::now();
        lantern_create_index::create_usearch_index(
            &CreateIndexArgs {
                import: true,
                out: index_path.clone(),
                table: args.table.clone(),
                schema: args.schema.clone(),
                metric_kind: args.metric_kind.clone(),
                efc: best_result.ef_construction as usize,
                ef: best_result.ef as usize,
                m: best_result.m as usize,
                uri: uri.clone(),
                column: args.column.clone(),
                dims: column_dims as usize,
                index_name: None,
            },
            Some(Logger::new(&logger.label, LogLevel::Info)),
            None,
        )?;
        let duration = start.elapsed().as_secs();
        logger.debug(&format!("Index for job {job_id} created in {duration}s"));
    }

    report_progress(&progress_cb, &logger, 100);
    Ok(())
}

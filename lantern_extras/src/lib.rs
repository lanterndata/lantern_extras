use pgrx::prelude::*;

use flate2::read::GzDecoder;
use ftp::FtpStream;
use lantern_embeddings_core::{
    self,
    core::{get_runtime, Runtime},
};
use lantern_external_index::cli::{CreateIndexArgs, UMetricKind};
use rand::Rng;
use tar::Archive;

pgrx::pg_module_magic!();
pub mod dotvecs;

static ORT_RUNTIME_PARAMS: &'static str = "{ \"cache\": true }";

fn notice_fn(text: &str) {
    notice!("{}", text);
}

fn validate_index_param(param_name: &str, param_val: i32, min: i32, max: i32) {
    if param_val < min || param_val > max {
        error!("{param_name} should be in range [{min}, {max}]");
    }
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

    let res = lantern_external_index::create_usearch_index(
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
    use crate::lantern_create_external_index;
    use pgrx::prelude::*;
    use pgrx::{PgBuiltInOids, PgRelation, Spi};

    #[pg_extern(immutable, parallel_unsafe)]
    fn _reindex_external_index<'a>(
        index: PgRelation,
        metric_kind: &'a str,
        dim: i32,
        m: i32,
        ef_construction: i32,
        ef: i32,
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
            &index_name,
        )
    }
}

#[pg_extern(immutable, parallel_safe)]
fn text_embedding<'a>(model_name: &'a str, text: &'a str) -> Result<Vec<f32>, anyhow::Error> {
    let runtime = get_runtime(
        &Runtime::Ort,
        Some(&(notice_fn as lantern_embeddings_core::core::LoggerFn)),
        ORT_RUNTIME_PARAMS.clone(),
    )?;
    let mut res = runtime.process(model_name, &vec![text])?;
    Ok(res.pop().unwrap())
}

#[pg_extern(immutable, parallel_safe)]
fn clip_text<'a>(text: &'a str) -> Result<Vec<f32>, anyhow::Error> {
    text_embedding("clip/ViT-B-32-textual", text)
}

#[pg_extern(immutable, parallel_safe)]
fn image_embedding<'a>(
    model_name: &'a str,
    path_or_url: &'a str,
) -> Result<Vec<f32>, anyhow::Error> {
    text_embedding(model_name, path_or_url)
}

#[pg_extern(immutable, parallel_safe)]
fn clip_image<'a>(path_or_url: &'a str) -> Result<Vec<f32>, anyhow::Error> {
    image_embedding("clip/ViT-B-32-visual", path_or_url)
}

#[pg_extern(immutable, parallel_safe)]
fn get_available_models() -> Result<String, anyhow::Error> {
    let runtime = get_runtime(
        &Runtime::Ort,
        Some(&(notice_fn as lantern_embeddings_core::core::LoggerFn)),
        ORT_RUNTIME_PARAMS.clone(),
    )?;
    return Ok(runtime.get_available_models().0);
}

#[pg_extern]
fn get_vectors<'a>(gzippath: &'a str) -> String {
    let url = url::Url::parse(gzippath).unwrap();
    if url.scheme() == "ftp" {
        match download_gzipped_ftp(url) {
            Ok(data) => {
                return data
                    .map(|b| b.unwrap().to_string())
                    .take(10)
                    .collect::<Vec<String>>()
                    .join(" ");
            }
            Err(e) => {
                return e.to_string();
            }
        }
    }
    return "not supported".to_string();
}

fn download_gzipped_ftp(
    url: url::Url,
) -> Result<impl Iterator<Item = Result<u8, std::io::Error>>, Box<dyn std::error::Error>> {
    use std::io::prelude::*;
    assert!(url.scheme() == "ftp");
    let domain = url.host_str().expect("no host");
    let port = url.port().unwrap_or(21);
    let pathurl = url.join("./")?;
    let path = pathurl.path();
    let filename = url
        .path_segments()
        .expect("expected path segments in an ftp url")
        .last()
        .unwrap();

    let mut ftp_stream = FtpStream::connect(format!("{}:{}", domain, port))?;
    ftp_stream
        .login("anonymous", "anonymous")
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::PermissionDenied, e.to_string()))?;
    ftp_stream.cwd(path)?;
    let file = ftp_stream.get(filename)?;

    let dd = GzDecoder::new(file);
    if false {
        return Ok(dd.bytes());
    }
    let mut a = Archive::new(dd);
    // a.unpack("/tmp/rustftp")?;
    a.entries()
        .unwrap()
        .map(|entry| match entry {
            Ok(e) => {
                let s = String::new();
                notice!("entry name {}", e.path().unwrap().display());
                Ok(s)
            }
            Err(e) => Err(e),
        })
        .for_each(|e| match e {
            Ok(s) => {
                notice!("entry: {}", s);
            }
            Err(e) => {
                notice!("entry: {}", e);
            }
        });
    return Err("not implemented".into());
}

#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {
        // perform one-off initialization when the pg_test framework starts
    }

    pub fn postgresql_conf_options() -> Vec<&'static str> {
        // return any postgresql.conf settings that are required for your tests
        vec![]
    }
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
pub mod tests {
    use crate::*;

    static HELLO_WORLD_TEXT: &'static str = "Hello world!";
    #[rustfmt::skip]
    static HELLO_WORLD_CLIP_EMB: &'static [f32] = &[-0.020476775, 0.111073, -0.10530874, 0.19828044, -0.027562656, -0.23474982, 0.21065629, -1.4416628, 0.14605063, -0.160044, -0.37608588, -0.2119374, -0.15697962, 0.008458294, 0.2106666, -0.017877355, 0.33919975, 0.045852974, -0.100722946, -0.045923192, 0.10990898, -0.25829625, 0.30317736, 0.11488654, -0.46186274, -0.4626967, -0.042635046, 0.29741955, -0.037274115, 0.27366883, 0.20929608, -0.16471854, 0.055385463, -0.24945396, 0.046744138, 0.23008388, -0.187143, -0.029809874, 0.07640592, -0.16700715, -0.085975975, -0.25384986, -0.11262718, -0.13075642, 0.14013499, -0.08889471, 0.063620545, -0.03663072, 0.07881963, 0.229372, 0.29210842, -0.31412473, -0.0078749135, -0.28392997, 0.15476987, -0.02043429, -0.06290087, -0.018591419, -0.08700596, -0.19504404, 0.44238496, -0.109749734, 0.275388, -0.05582391, 0.13863862, -0.033968866, 0.19097951, -0.13578185, 0.14519869, 0.045676023, 0.25584215, -0.015645579, -0.26207292, 0.24827223, 0.03445548, -0.31498873, 0.2824291, 0.56327266, 0.013674453, -0.16007274, -0.21102041, -0.19478007, -0.14121902, -0.16792865, -0.40819037, 0.0438246, 0.22567344, -0.3297562, 0.11681607, 0.11226921, 0.003411904, 0.0825413, -1.9214885, -0.20295206, -0.01407405, -0.12056728, -0.10185087, 0.2004847, -0.10759242, -0.19343027, 0.23292445, 0.33320382, 0.22149219, -0.08638317, 0.23560561, -0.19066495, 0.17755258, -0.16993555, -0.029849865, -0.22932798, 0.08929604, 0.03428103, -0.107722655, -0.024028901, 0.1847432, 0.09379715, 0.068997346, 0.14411347, 0.07746827, 0.04011213, 0.0658475, 0.20222189, -0.030440576, 0.08776434, -0.13246597, 0.016684677, 0.047582008, 0.05198705, 0.0700707, 0.084653914, 0.06909941, 0.10376939, -0.081172146, 7.3704977, -0.14797403, 0.21361199, -0.08800018, -0.35132965, 0.021939317, -0.09089277, 0.037140936, -0.35118827, -0.04941287, -0.026789397, -0.23948334, 0.10874768, -0.26285917, 0.36523858, 0.50445855, 0.10481266, 0.4098543, -0.3129306, -0.59758216, -0.034414284, 0.027843453, -0.091017574, -0.32560846, 0.30377156, -0.23118782, 0.065180466, -0.36710367, 0.012223348, -0.16262382, 0.13480008, 0.13736263, 0.0075045526, 0.23728539, 0.2116946, 0.3838804, 0.14245796, 0.029927406, -0.1341641, -0.03992462, -0.11786532, -0.30819097, -0.12132531, -0.30330336, -0.12580931, -0.035588562, 0.16817878, -0.038514666, 0.18975018, 0.024216771, 0.026217014, -0.090409905, -0.07525094, 0.27214733, 0.044363648, 0.22049513, -0.16281407, 0.0631836, -0.053991392, 0.22901943, 0.060611814, 0.32349586, 0.012564644, -0.08271384, -0.078071356, -0.040050197, 0.30019575, -0.020119771, -0.24983557, -0.27131745, -0.054717228, 0.10751669, -0.22137052, -0.013891213, 0.1374619, 0.010636002, 0.14240754, 0.15695044, 0.13009356, 0.09398372, -0.025530757, -0.23091945, 0.025572032, 0.0031829178, -0.24976723, 0.21949057, -0.27335533, 0.10732825, -0.06507754, -0.13964619, -0.12820473, 0.0877506, -0.12037073, -0.09331375, 0.5416604, 0.09421308, 0.10336839, 0.13041648, 0.028918259, -0.20625801, 0.43129992, 0.1459057, -0.016492277, 0.04081285, 0.047613487, 0.009934774, 0.009535242, 0.24991985, 0.30162302, -0.035831526, -0.486154, -0.05103448, 0.6563518, 0.009447038, -0.46683514, -0.08313032, 0.03696554, 0.14653865, -0.18361577, -0.050656207, -0.22145961, -0.24882938, 0.16182947, 0.038483176, 0.021601643, 0.21019635, -0.08234185, 0.2780598, -0.2908701, 0.02953285, 0.12296721, -0.21878204, -0.128144, -0.060083054, -0.24550691, -0.0879209, 0.07546242, -0.07249651, 0.020485193, -0.03692793, 0.16765073, -0.09103814, 0.23033753, -0.21836475, 0.1664663, -0.21737409, 0.16596827, 0.31133112, -0.10023908, 0.3706924, -0.2464955, 0.24176213, 0.32084322, 0.16259453, -0.05540514, 0.05083946, -0.12935856, 0.03267494, 0.18455097, 0.011275742, 0.101560265, -0.03561543, 0.27086422, -0.09041187, -0.3287123, 0.040132664, -0.1831892, 0.0047493726, -0.033484608, 0.19651815, -0.31473777, 0.13112763, 0.3413608, -0.2455968, 0.045542836, -0.104461126, 0.005065279, 0.4251323, -0.23926145, 7.362257, -0.23300058, -0.007646084, 0.045260623, -0.04437712, -0.15608308, 0.10616787, 0.07287562, 0.020465162, 0.5783111, -0.09140764, 0.13351701, -0.10439435, -0.25910264, -0.24776801, -0.11816393, -0.0561099, -2.770246, -0.31068143, 0.14091884, 0.070470706, -0.058813483, -0.024911582, -0.15688656, 0.21985042, 0.36713743, 0.107128955, 0.13669589, 0.0051332787, 0.23876746, 0.40614, 0.060822293, -0.33256605, 0.08021091, -0.07582166, -0.23415035, -0.04432006, -0.07010134, -0.22593829, -0.065982476, 0.0427081, 0.028756332, -0.060415134, 0.4012118, 0.04000116, 0.38686636, 0.15789485, -0.3262711, 0.00475822, 0.1094888, 0.57392967, 0.34066314, -0.20665756, -0.057930544, 0.22794127, 0.15457465, -0.18280812, 0.09717287, -0.059893936, 0.6086165, -0.21629295, 0.23789874, -0.29008394, 0.20201065, -0.32359564, 0.10615185, -0.27292472, -0.07158633, -0.28011638, -0.34555736, 0.06371697, -0.08473098, 0.06386079, 0.056933045, -0.22559868, -0.14229044, -0.28497994, 0.12396638, -0.22131008, -0.5294925, -0.0767176, 0.11793665, 0.057084702, 0.0040413737, -0.06450936, 0.01728496, -0.3512309, 0.3330403, -0.06519018, 0.20660913, 0.05178895, 0.343586, 0.20610496, 0.15943015, -0.20400554, 0.030726567, 0.30282435, -0.010365292, 0.09665407, 0.14873841, 0.25734085, 0.21442415, 0.40125173, -0.08738096, -0.06913735, 0.2987156, 0.27425864, -0.00093972683, 0.04379954, 0.038295284, 0.06546015, -0.32040307, -0.20215109, 0.41629654, 0.047880255, -0.21194798, -0.036038153, 0.1522459, -0.19484445, 0.039462823, -0.21064907, -0.13694683, -0.2501285, 0.082520045, -0.19627452, -0.4384591, 0.23384215, 0.109192975, 0.15369438, -0.27861565, 0.11628551, -0.040622413, -0.04433043, -0.12166405, 0.13092265, -0.07827225, 0.11547213, -0.030894607, -0.18117768, 0.020442963, -0.17173964, 0.2974965, 0.32407364, 0.04766877, 0.10717582, 0.09193277, -0.16506508, 0.13757893, 0.096764445, -0.5020385, -0.016144186, 0.019135457, -0.29343155, -0.100692585, 0.37422845, 0.022318047, -0.15931854, 0.017905466, -0.13255252, 0.109108776, 0.0065737963, -0.074496314, 0.16018367, 0.16922836, 0.10337954, 0.14559221, 0.12458767, 0.22240283, -0.028031524, -0.80666745, -0.12128662, 0.01321203, -0.3255096, -0.06868529, -0.19293119, -0.025735162, -0.28654903, 0.020253882, -0.38534135, -0.038486935, 0.06613082, -0.06418133, -0.15839031, 0.06471726, -0.19283788, -0.34131792, 0.15264772, -0.23320137, -0.089332774, -0.041326385, 0.13021827, 0.102788255, -0.022895776, 0.09561402, 0.10994218, -0.032236956, -0.024895646, -0.2531594, -0.27954918, 0.23604941];

    #[pg_test]
    fn test_clip_text() {
        let embedding =
            Spi::get_one::<Vec<f32>>(&format!("SELECT clip_text('{HELLO_WORLD_TEXT}');")).unwrap();
        assert_eq!(embedding.unwrap(), HELLO_WORLD_CLIP_EMB.to_vec())
    }
}

use clap::Parser;
use lantern_create_index::cli::UMetricKind;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct IndexAutotuneArgs {
    /// Fully associated database connection string including db name
    #[arg(short, long)]
    pub uri: String,

    /// Schema name
    #[arg(short, long, default_value = "public")]
    pub schema: String,

    /// Table name
    #[arg(short, long)]
    pub table: String,

    /// Column name
    #[arg(short, long)]
    pub column: String,

    /// Primary key name
    #[arg(long)]
    pub pk: String,

    /// Target recall
    #[arg(long, default_value_t = 98)]
    pub recall: u64,

    /// Target recall
    #[arg(long, default_value_t = 10)]
    pub k: u16,

    /// Distance algorithm
    #[arg(long, value_enum, default_value_t = UMetricKind::L2sq)]
    pub metric_kind: UMetricKind,
}

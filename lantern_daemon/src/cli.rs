use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct DaemonArgs {
    /// Fully associated database connection string including db name to get jobs
    #[arg(short, long)]
    pub uri: String,

    /// Jobs table name
    #[arg(short, long)]
    pub table: String,

    /// Schema name
    #[arg(short, long, default_value = "public")]
    pub schema: String,

    /// Max concurrent jobs
    #[arg(short, long, default_value_t = 1)]
    pub queue_size: usize,
}

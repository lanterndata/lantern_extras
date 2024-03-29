mod autotune_jobs;
pub mod cli;
mod client_embedding_jobs;
mod embedding_jobs;
mod external_index_jobs;
mod helpers;
mod types;

use std::sync::{
    mpsc::{self, Receiver, Sender},
    Arc,
};

use crate::logger::Logger;
use crate::types::AnyhowVoidResult;

pub fn start(
    args: cli::DaemonArgs,
    logger: Option<Logger>,
    error_channel: Option<(Sender<String>, Receiver<String>)>,
) -> AnyhowVoidResult {
    let logger = Arc::new(logger.unwrap_or(Logger::new("Lantern Daemon", args.log_level.value())));
    let (error_sender, error_receiver) = error_channel.unwrap_or(mpsc::channel());

    let embedding_args = args.clone();
    let autotune_args = args.clone();
    let external_index_args = args.clone();
    let embedding_logger = Arc::new(Logger::new(
        "Lantern Daemon Embeddings",
        logger.level.clone(),
    ));
    let autotune_logger = Arc::new(Logger::new("Lantern Daemon Autotune", logger.level.clone()));
    let external_index_logger = Arc::new(Logger::new(
        "Lantern Daemon External Index",
        logger.level.clone(),
    ));

    logger.info("Starting Daemon");

    if args.embedding_table.is_some() {
        let embedding_error_sender = error_sender.clone();
        std::thread::spawn(move || {
            if let Err(e) = embedding_jobs::start(embedding_args, embedding_logger) {
                embedding_error_sender
                    .send(format!("Embeddings Error: {e}"))
                    .unwrap();
            }
        });
    }

    if args.autotune_table.is_some() {
        let autotune_error_sender = error_sender.clone();
        std::thread::spawn(move || {
            if let Err(e) = autotune_jobs::start(autotune_args, autotune_logger) {
                autotune_error_sender
                    .send(format!("Autotune Error: {e}"))
                    .unwrap();
            }
        });
    }

    if args.external_index_table.is_some() {
        let external_index_error_sender = error_sender.clone();
        std::thread::spawn(move || {
            if let Err(e) = external_index_jobs::start(external_index_args, external_index_logger) {
                external_index_error_sender
                    .send(format!("External Index Error: {e}"))
                    .unwrap();
            }
        });
    }

    if let Ok(err_msg) = error_receiver.recv() {
        anyhow::bail!("{err_msg}");
    }

    Ok(())
}

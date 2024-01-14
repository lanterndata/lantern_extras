use std::str::FromStr;

use crate::{ort_runtime::OrtRuntime, runtime::EmbeddingRuntime};

fn default_logger(text: &str) {
    println!("{}", text);
}

#[derive(Debug, PartialEq, Clone)]
pub enum Runtime {
    Ort,
}

pub type LoggerFn = fn(&str);
impl FromStr for Runtime {
    type Err = anyhow::Error;
    fn from_str(input: &str) -> Result<Runtime, anyhow::Error> {
        match input {
            "ort" => Ok(Runtime::Ort),
            _ => anyhow::bail!("Invalid runtime {input}"),
        }
    }
}

impl ToString for Runtime {
    fn to_string(&self) -> String {
        match self {
            Runtime::Ort => "ort".to_owned(),
        }
    }
}

pub fn get_runtime<'a>(
    runtime: &Runtime,
    logger: Option<&'a LoggerFn>,
    params: &'a str,
) -> Result<Box<dyn EmbeddingRuntime + 'a>, anyhow::Error> {
    let embedding_runtime = match runtime {
        Runtime::Ort => OrtRuntime::new(logger.unwrap_or(&(default_logger as LoggerFn)), params),
    };

    Ok(Box::new(embedding_runtime?))
}

pub fn get_available_runtimes() -> Vec<String> {
    todo!();
}

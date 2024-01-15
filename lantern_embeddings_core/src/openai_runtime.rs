use std::{collections::HashMap, sync::RwLock};

use crate::{core::LoggerFn, runtime::EmbeddingRuntime, HTTPRuntime};
use serde::Deserialize;
use tiktoken_rs::{cl100k_base, CoreBPE};

struct ModelInfo {
    tokenizer: CoreBPE,
    sequence_len: usize,
    dimensions: usize,
}

#[derive(Deserialize)]
struct OpenAiEmbedding {
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct OpenAiResponse {
    data: Vec<OpenAiEmbedding>,
}

impl ModelInfo {
    pub fn new(model_name: &str) -> Result<Self, anyhow::Error> {
        match model_name {
            "text-embedding-ada-002" => Ok(Self {
                tokenizer: cl100k_base()?,
                sequence_len: 8192,
                dimensions: 1536,
            }),
            _ => anyhow::bail!("Unsupported model {model_name}"),
        }
    }
}

lazy_static! {
    static ref MODEL_INFO_MAP: RwLock<HashMap<&'static str, ModelInfo>> =
        RwLock::new(HashMap::from([(
            "text-embedding-ada-002",
            ModelInfo::new("text-embedding-ada-002").unwrap()
        ),]));
}

pub struct OpenAiRuntime<'a> {
    request_timeout: u64,
    base_url: String,
    headers: Vec<(String, String)>,
    #[allow(dead_code)]
    logger: &'a LoggerFn,
}

#[derive(Deserialize)]
pub struct OpenAiRuntimeParams {
    api_token: Option<String>,
}

impl<'a> OpenAiRuntime<'a> {
    pub fn new(logger: &'a LoggerFn, params: &'a str) -> Result<Self, anyhow::Error> {
        let runtime_params: OpenAiRuntimeParams = serde_json::from_str(&params)?;

        if runtime_params.api_token.is_none() {
            anyhow::bail!("'api_token' is required for OpenAi runtime");
        }

        Ok(Self {
            base_url: "https://api.openai.com".to_owned(),
            logger,
            request_timeout: 120,
            headers: vec![
                ("Content-Type".to_owned(), "application/json".to_owned()),
                (
                    "Authorization".to_owned(),
                    format!("Bearer {}", runtime_params.api_token.unwrap()),
                ),
            ],
        })
    }

    fn group_vectors_by_token_count(
        &self,
        input: Vec<Vec<usize>>,
        max_token_count: usize,
    ) -> Vec<Vec<Vec<usize>>> {
        let mut result = Vec::new();
        let mut current_group = Vec::new();
        let mut current_group_token_count = 0;

        for inner_vec in input {
            let inner_vec_token_count = inner_vec.len();

            if current_group_token_count + inner_vec_token_count <= max_token_count {
                // Add the inner vector to the current group
                current_group.push(inner_vec);
                current_group_token_count += inner_vec_token_count;
            } else {
                // Start a new group
                result.push(current_group);
                current_group = vec![inner_vec];
                current_group_token_count = inner_vec_token_count;
            }
        }

        // Add the last group if it's not empty
        if !current_group.is_empty() {
            result.push(current_group);
        }

        result
    }

    fn chunk_inputs(
        &self,
        model_name: &str,
        inputs: &Vec<&str>,
    ) -> Result<Vec<String>, anyhow::Error> {
        let model_map = MODEL_INFO_MAP.read().unwrap();
        let model_info = model_map.get(model_name);

        if model_info.is_none() {
            anyhow::bail!("Unsupported model {model_name}");
        }

        let model_info = model_info.unwrap();
        let token_groups: Vec<Vec<usize>> = inputs
            .iter()
            .map(|input| {
                let mut tokens = model_info.tokenizer.encode_with_special_tokens(input);
                if tokens.len() > model_info.sequence_len {
                    tokens.truncate(model_info.sequence_len);
                }
                tokens
            })
            .collect();

        let batch_tokens: Vec<String> = self
            .group_vectors_by_token_count(token_groups, model_info.sequence_len)
            .iter()
            .map(|token_group| {
                let json_string = serde_json::to_string(token_group).unwrap();
                format!(
                    r#"
                 {{
                   "input": {json_string},
                   "model": "{model_name}"
                 }}
                "#
                )
            })
            .collect();

        Ok(batch_tokens)
    }

    pub fn get_response(&self, body: Vec<u8>) -> Result<Vec<Vec<f32>>, anyhow::Error> {
        let result: Result<OpenAiResponse, serde_json::Error> = serde_json::from_slice(&body);
        if let Err(e) = result {
            anyhow::bail!(
                "Error: {e}. OpenAI response : {:?}",
                serde_json::from_slice::<serde_json::Value>(&body)?
            );
        }

        let result = result.unwrap();

        Ok(result
            .data
            .iter()
            .map(|emb| emb.embedding.clone())
            .collect())
    }
}

impl<'a> EmbeddingRuntime for OpenAiRuntime<'a> {
    fn process(
        &self,
        model_name: &str,
        inputs: &Vec<&str>,
    ) -> Result<Vec<Vec<f32>>, anyhow::Error> {
        self.post_request("/v1/embeddings", model_name, inputs)
    }

    fn get_available_models(&self) -> (String, Vec<(String, bool)>) {
        let map = MODEL_INFO_MAP.read().unwrap();
        let mut res = String::new();
        let mut models = Vec::with_capacity(map.len());
        for (key, value) in &*map {
            res.push_str(&format!("{} - sequence_len: {}, dimensions: {}\n", key, value.sequence_len, value.dimensions));
            models.push((key.to_string(), false));
        }

        return (res, models);
    }
}
HTTPRuntime!(OpenAiRuntime);

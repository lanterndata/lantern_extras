use std::{fs, path::Path};

pub fn bytes_to_integer_le<T>(bytes: &[u8]) -> T
where
    T: From<u8> + std::ops::Shl<usize, Output = T> + std::ops::BitOr<Output = T> + Default,
{
    let mut result: T = Default::default();

    for &byte in bytes.iter().rev() {
        result = (result << 8) | T::from(byte);
    }

    result
}

pub fn create_index_dir(data_dir: &str) -> Result<String, anyhow::Error> {
    let path = Path::new(data_dir).join("ldb_indexes");
    fs::create_dir_all(&path)?;
    Ok(path.to_str().unwrap().to_owned())
}

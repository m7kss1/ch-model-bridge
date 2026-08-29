use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Embedding,
    Rerank,
    /// Arbitrary ONNX model over numeric features: one float-tensor input of
    /// shape [N, n_features], one float output of shape [N] or [N, 1]. No
    /// tokenizer; `tokenizer.json` is not part of the passport.
    Tabular,
}

/// Model passport: the only way a model enters the daemon. The checksums cover
/// every file the runtime will load, so a silently replaced `model.onnx` or a
/// tokenizer from another model is rejected instead of degrading results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Passport {
    pub name: String,
    pub kind: ModelKind,
    /// Model directory; a relative path is resolved against the passport file
    /// location, so a passport tree can be moved as a whole.
    pub dir: PathBuf,
    pub revision: u32,
    #[serde(default = "default_max_batch")]
    pub max_batch: usize,
    pub sha256: BTreeMap<String, String>,
}

fn default_max_batch() -> usize {
    64
}

impl Passport {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| path.display().to_string())?;
        let passport: Passport = toml::from_str(&text)
            .with_context(|| format!("{}: invalid passport", path.display()))?;
        if passport.sha256.is_empty() {
            bail!("{}: passport lists no files to verify", path.display());
        }
        Ok(passport)
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let text = toml::to_string_pretty(self).context("serialize passport")?;
        std::fs::write(path, text).with_context(|| path.display().to_string())?;
        Ok(())
    }

    pub fn resolved_dir(&self, passport_path: &Path) -> PathBuf {
        if self.dir.is_absolute() {
            self.dir.clone()
        } else {
            passport_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(&self.dir)
        }
    }

    /// Verifies every listed file against its recorded checksum. Fails closed:
    /// a missing file or a mismatch refuses the model instead of loading what
    /// happens to be on disk.
    pub fn verify(&self, dir: &Path) -> anyhow::Result<()> {
        for (file, expected) in &self.sha256 {
            let path = dir.join(file);
            let actual = sha256_file(&path)?;
            if &actual != expected {
                bail!(
                    "{}: checksum mismatch (expected {expected}, got {actual}); \
                     the file differs from the one this passport was issued for",
                    path.display()
                );
            }
        }
        Ok(())
    }
}

pub fn sha256_file(path: &Path) -> anyhow::Result<String> {
    use sha2::{Digest, Sha256};

    let mut file = std::fs::File::open(path).with_context(|| path.display().to_string())?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    let digest = hasher.finalize();
    Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
}

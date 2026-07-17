use serde::{Deserialize, Serialize};

/// Persistent identifier for a layer. Stable across saves and reorders — do
/// not derive from position in `Graph::layers`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LayerId(pub u64);

impl std::fmt::Display for LayerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "L{}", self.0)
    }
}

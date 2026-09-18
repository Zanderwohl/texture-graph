//! Small graph-adjacent helpers shared by panels.

use texture_graph_core::Graph;

/// First name in `base`, `base 1`, `base 2`, … not taken by any layer.
pub fn unique_name(graph: &Graph, base: &str) -> String {
    if !graph.layers.iter().any(|l| l.name == base) {
        return base.to_string();
    }
    for n in 1..u32::MAX {
        let candidate = format!("{base} {n}");
        if !graph.layers.iter().any(|l| l.name == candidate) {
            return candidate;
        }
    }
    base.to_string()
}

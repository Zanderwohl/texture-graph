//! Sampling a graph on a sphere rather than a plane.
//!
//! A sphere bake samples the sphere inscribed in the unit cube (center
//! `(0.5, 0.5, 0.5)`, radius 0.5), so it matches a volume bake restricted to
//! that shell. A cubemap stores that field at higher resolution than a volume
//! of the same memory, since it stores no interior.
//!
//! Faces follow the WebGPU/Vulkan/D3D cube layout: layer order
//! `+X, -X, +Y, -Y, +Z, -Z`, and within a face `u` runs right and `v` runs
//! down. A mistake here mirrors or rotates faces without failing anything, so
//! `bake_test` checks it against the GPU's cube sampler.

use crate::eval::Sample;

pub const CUBE_FACES: u32 = 6;

/// The unnormalized direction a cube face's `(u, v)` looks along, with
/// `u, v ∈ [0, 1]`. The major axis has magnitude one. Panics if `face >= 6`.
pub fn cube_direction(face: u32, u: f32, v: f32) -> [f32; 3] {
    let s = 2.0 * u - 1.0;
    let t = 2.0 * v - 1.0;
    match face {
        0 => [1.0, -t, -s],
        1 => [-1.0, -t, s],
        2 => [s, 1.0, t],
        3 => [s, -1.0, -t],
        4 => [s, -t, 1.0],
        5 => [-s, -t, -1.0],
        _ => panic!("a cube has six faces; asked for face {face}"),
    }
}

/// Where a sphere bake samples the graph for a face's `(u, v)`: that
/// direction's point on the sphere inscribed in the unit cube.
pub fn cube_sample(face: u32, u: f32, v: f32) -> Sample {
    let [x, y, z] = cube_direction(face, u, v);
    let r = (x * x + y * y + z * z).sqrt();
    Sample::new(x / r * 0.5 + 0.5, y / r * 0.5 + 0.5, z / r * 0.5 + 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Face centers point along the axes in layer order.
    #[test]
    fn face_centers_are_the_six_axes() {
        let axes = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        for (face, axis) in axes.iter().enumerate() {
            assert_eq!(&cube_direction(face as u32, 0.5, 0.5), axis, "face {face}");
        }
    }

    /// Every face edge is shared with another face. A mirrored face would
    /// pass the center test and fail this.
    #[test]
    fn edges_are_shared_between_faces() {
        let key = |d: [f32; 3]| d.map(|c| (c * 1000.0).round() as i32);
        let mut edges = std::collections::HashMap::new();
        for face in 0..CUBE_FACES {
            for k in 0..=8 {
                let a = k as f32 / 8.0;
                for (u, v) in [(a, 0.0), (a, 1.0), (0.0, a), (1.0, a)] {
                    edges.entry(key(cube_direction(face, u, v))).or_insert_with(Vec::new).push(face);
                }
            }
        }
        for (dir, faces) in edges {
            let mut distinct = faces.clone();
            distinct.sort();
            distinct.dedup();
            // Corners belong to three faces, the rest of an edge to two.
            assert!(distinct.len() >= 2, "direction {dir:?} lies on one face only: {faces:?}");
        }
    }

    #[test]
    fn samples_lie_on_the_inscribed_sphere() {
        for face in 0..CUBE_FACES {
            for (u, v) in [(0.0, 0.0), (0.3, 0.8), (1.0, 0.5)] {
                let s = cube_sample(face, u, v);
                let r = ((s.u - 0.5).powi(2) + (s.v - 0.5).powi(2) + (s.w - 0.5).powi(2)).sqrt();
                assert!((r - 0.5).abs() < 1e-6, "face {face} ({u}, {v}) at radius {r}");
            }
        }
    }
}

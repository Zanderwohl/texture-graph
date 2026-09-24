//! Crater fields: the specification both backends implement.
//!
//! `crates/gpu/src/shaders/craters.wgsl` mirrors this file function for
//! function. Change both in the same commit.
//!
//! # Scatter
//!
//! Each size class is a lattice of cells, twice as fine as the class before.
//! A cell holds at most one crater, at a hashed point inside it. A crater is
//! seen only from the slab `|h| < 1/2` cell about the surface, `h` measured
//! along the surface normal, so every orientation of the surface against the
//! lattice sees one layer of centers. Its extent is at most half a cell
//! across the slab, so every crater that can reach a sample lies in the 27
//! cells around it and the field is continuous across cell faces.
//!
//! Distance is measured in the tangent plane, so a crater is round on a
//! sphere rather than a slice through a ball.
//!
//! # Composition
//!
//! A crater pulls what it lands on toward [`DATUM`] by [`Spec::erase`] inside
//! its rim, and less across its ejecta, then adds its own relief. Classes run
//! largest first, so small craters sit on large ones; within a class the
//! order is by cell. Chaining nodes through `under` is how one series of
//! impacts overprints an older one.
//!
//! # Parity
//!
//! The hash is all `u32` and matches bit for bit. Every cut is soft, so a
//! rounding difference moves the output by rounding, never by a whole crater.

/// Upper bound on [`Spec::classes`].
pub const MAX_CLASSES: u32 = 6;

/// The height a crater erases toward. Also a flat surface's height.
pub const DATUM: f32 = 0.5;

/// Largest rim radius, in cells. With [`EXTENT`] it keeps a crater within
/// half a cell.
const R_MAX: f32 = 0.2;

/// How far the ejecta reaches, in rim radii.
const EXTENT: f32 = 2.5;

/// Rays per crater.
const RAYS: u32 = 6;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Spec {
    /// Cells per unit of sample space in the largest class.
    pub frequency: f32,
    /// `1..=`[`MAX_CLASSES`].
    pub classes: u32,
    /// Share of cells occupied, per class, relative to the class before. At 1
    /// the cumulative count goes as `D^-2`; above it small craters dominate.
    pub gain: f32,
    /// Bowl depth over rim radius for a fresh simple crater.
    pub depth: f32,
    /// `[0, 1]`: shallows the bowl, lowers the rim and fades the ejecta.
    pub age: f32,
    /// `[0, 1]`: how far a crater erases what was under it.
    pub erase: f32,
    /// Central peak height in bowl depths, on the largest craters only.
    pub peak: f32,
    /// `[0, 1]`: how much of the ejecta's brightness is in rays.
    pub rays: f32,
    /// Heights in sample-space units times this.
    pub relief: f32,
    /// Normals point away from the unit cube's center, as on a sphere bake.
    /// Otherwise they are `+w`, as on a flat one.
    pub sphere: bool,
    /// Ejecta brightness in `[0, 1]` instead of height.
    pub ejecta: bool,
}

pub fn pcg(v: u32) -> u32 {
    let s = v.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let w = ((s >> ((s >> 28) + 4)) ^ s).wrapping_mul(277_803_737);
    (w >> 22) ^ w
}

/// `[0, 1)`, exact in `f32`.
pub fn unit(h: u32) -> f32 {
    (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

fn cell_hash(seed: u32, class: u32, c: [i32; 3]) -> u32 {
    let h = pcg(seed ^ class.wrapping_mul(0x9E37_79B9));
    let h = pcg(h.wrapping_add(c[0] as u32));
    let h = pcg(h.wrapping_add(c[1] as u32));
    pcg(h.wrapping_add(c[2] as u32))
}

fn saturate(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Brightness of the ejecta toward `dir`, a unit tangent: `RAYS` narrow lobes
/// about hashed directions, summed.
fn ray_lobes(h: u32, dir: [f32; 3]) -> f32 {
    let mut h = h;
    let mut sum = 0.0;
    for _ in 0..RAYS {
        h = pcg(h);
        let a = unit(h) * 2.0 - 1.0;
        h = pcg(h);
        let b = unit(h) * 2.0 - 1.0;
        h = pcg(h);
        let c = unit(h) * 2.0 - 1.0;
        let len = (a * a + b * b + c * c).sqrt().max(1.0e-3);
        // Out of the tangent plane a direction shortens, which narrows its lobe
        // further: variety for nothing.
        let mut l = saturate(dot(dir, [a / len, b / len, c / len]));
        for _ in 0..5 {
            l *= l;
        }
        sum += l;
    }
    saturate(sum)
}

/// The field at `p`, a point in sample space, over `under`, with `density`
/// the share of the largest class's cells occupied.
pub fn sample(spec: &Spec, seed: u32, p: [f32; 3], under: f32, density: f32) -> f32 {
    let n = if spec.sphere {
        let d = [p[0] - 0.5, p[1] - 0.5, p[2] - 0.5];
        let len = dot(d, d).sqrt().max(1.0e-6);
        [d[0] / len, d[1] / len, d[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    };
    let mut out = under;
    let mut frequency = spec.frequency;
    let mut occupancy = density;
    let mut shrink = 1.0;
    for class in 0..spec.classes.clamp(1, MAX_CLASSES) {
        let q = [p[0] * frequency, p[1] * frequency, p[2] * frequency];
        let base = [q[0].floor() as i32, q[1].floor() as i32, q[2].floor() as i32];
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let c = [base[0] + dx, base[1] + dy, base[2] + dz];
                    let mut h = cell_hash(seed, class, c);
                    // Soft, so a density between two craters' draws fades one in.
                    let present = saturate((occupancy - unit(h)) * 20.0);
                    if present <= 0.0 {
                        continue;
                    }
                    h = pcg(h);
                    let cx = c[0] as f32 + unit(h);
                    h = pcg(h);
                    let cy = c[1] as f32 + unit(h);
                    h = pcg(h);
                    let cz = c[2] as f32 + unit(h);
                    let v = [q[0] - cx, q[1] - cy, q[2] - cz];
                    let across = dot(v, n);
                    let slab = saturate((0.5 - across.abs()) * 8.0);
                    if slab <= 0.0 {
                        continue;
                    }
                    h = pcg(h);
                    let size = 0.5 + 0.5 * unit(h);
                    let r = R_MAX * size;
                    let rho2 = (dot(v, v) - across * across).max(0.0);
                    let x = rho2.sqrt() / r;
                    if x >= EXTENT {
                        continue;
                    }
                    let weight = present * slab;
                    let t = (x - 1.0) / (EXTENT - 1.0);
                    // Wholly inside the rim, and fading across the blanket.
                    let cover = if x < 1.0 { 1.0 } else { (1.0 - t) * (1.0 - t) };
                    // 1 on the largest craters of the first class, 0 by the third.
                    let big = saturate((size * shrink - 0.3) / 0.5);
                    if spec.ejecta {
                        let tangent = [v[0] - across * n[0], v[1] - across * n[1], v[2] - across * n[2]];
                        let len = dot(tangent, tangent).sqrt().max(1.0e-6);
                        let dir = [tangent[0] / len, tangent[1] / len, tangent[2] / len];
                        let lobes = ray_lobes(pcg(h), dir);
                        let rim = mix(1.0, 0.35 + 0.65 * lobes, spec.rays);
                        let bright = if x < 1.0 {
                            let x2 = x * x;
                            mix(0.55, rim, x2 * x2)
                        } else {
                            rim
                        };
                        out = mix(out, bright * (1.0 - spec.age), weight * cover);
                    } else {
                        let d = spec.depth * (1.0 - 0.5 * big) * (1.0 - 0.75 * spec.age);
                        let rim_h = 0.3 * d * (1.0 - 0.4 * spec.age);
                        let z = if x < 1.0 {
                            let floor = rim_h - d * (1.0 - 0.6 * big);
                            let bowl = (d * (x * x - 1.0) + rim_h).max(floor);
                            let s = saturate(1.0 - (x * x) * 16.0);
                            bowl + spec.peak * big * d * s * s
                        } else {
                            let inv = 1.0 / (x * x * x);
                            const TAIL: f32 = 1.0 / (EXTENT * EXTENT * EXTENT);
                            rim_h * (inv - TAIL) / (1.0 - TAIL)
                        };
                        let erased = mix(out, DATUM, spec.erase * weight * cover);
                        // Rim radii are in cells; a cell is 1 / frequency.
                        out = erased + z * r / frequency * spec.relief * weight;
                    }
                }
            }
        }
        frequency *= 2.0;
        occupancy *= spec.gain;
        shrink *= 0.5;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> Spec {
        Spec {
            frequency: 4.0,
            classes: 4,
            gain: 1.0,
            depth: 0.4,
            age: 0.0,
            erase: 1.0,
            peak: 0.5,
            rays: 0.5,
            relief: 10.0,
            sphere: true,
            ejecta: false,
        }
    }

    /// Points on the sphere a sphere bake samples, spread by a golden spiral.
    fn sphere_points(n: usize) -> impl Iterator<Item = [f32; 3]> {
        (0..n).map(move |i| {
            let y = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
            let r = (1.0 - y * y).sqrt();
            let a = i as f32 * 2.399_963;
            [0.5 + 0.5 * r * a.cos(), 0.5 + 0.5 * y, 0.5 + 0.5 * r * a.sin()]
        })
    }

    #[test]
    fn no_craters_leaves_the_ground_alone() {
        for p in sphere_points(500) {
            assert_eq!(sample(&spec(), 7, p, 0.37, 0.0), 0.37);
        }
    }

    /// The field is a sum over the 27 cells about a point, so a crater that
    /// leaked past them would show as a step where the point crosses a cell
    /// face.
    #[test]
    fn crossing_a_cell_face_is_not_a_step() {
        for ejecta in [false, true] {
            let s = Spec { ejecta, classes: 2, ..spec() };
            let steps = 20_000;
            let mut worst: f32 = 0.0;
            let mut prev = None;
            for i in 0..steps {
                let t = i as f32 / steps as f32;
                // A great circle, crossing faces of every orientation.
                let a = t * std::f32::consts::TAU;
                let p = [0.5 + 0.5 * a.cos(), 0.5 + 0.35 * a.sin(), 0.5 + 0.357 * a.sin()];
                let v = sample(&s, 3, p, 0.5, 0.9);
                if let Some(u) = prev {
                    worst = worst.max((v - u as f32).abs());
                }
                prev = Some(v);
            }
            assert!(worst < 0.05, "ejecta {ejecta}: a step of {worst}");
        }
    }

    /// Where a later series lands it replaces the earlier one; elsewhere the
    /// earlier one shows through untouched.
    #[test]
    fn a_later_series_wipes_out_what_it_lands_on() {
        let old = Spec { frequency: 8.0, age: 0.6, ..spec() };
        let young = Spec { frequency: 3.0, classes: 2, ..spec() };
        let (mut wiped, mut kept) = (0, 0);
        for p in sphere_points(4000) {
            let before = sample(&old, 1, p, DATUM, 1.0);
            let over_old = sample(&young, 2, p, before, 0.5);
            let over_flat = sample(&young, 2, p, DATUM, 0.5);
            let (through, was) = ((over_old - over_flat).abs(), (before - DATUM).abs());
            assert!(through <= was + 1.0e-6, "erasing amplified the old relief: {through} > {was}");
            if was > 1.0e-3 && through < 0.01 * was {
                wiped += 1;
            }
            if was > 1.0e-3 && (through - was).abs() < 1.0e-6 {
                kept += 1;
            }
        }
        assert!(wiped > 40, "only {wiped} old points erased");
        assert!(kept > 100, "only {kept} old points untouched");
    }

    #[test]
    fn height_and_ejecta_mark_the_same_craters() {
        let mut both = 0;
        for p in sphere_points(3000) {
            let h = sample(&spec(), 5, p, DATUM, 0.7);
            let e = sample(&Spec { ejecta: true, ..spec() }, 5, p, 0.0, 0.7);
            assert!((0.0..=1.0).contains(&e), "ejecta {e} out of range");
            assert_eq!(h == DATUM, e == 0.0, "at {p:?}: height {h}, ejecta {e}");
            both += usize::from(e > 0.0);
        }
        assert!(both > 300, "{both}");
    }

    /// More density is more of the surface cratered, and an old series is
    /// shallower than a fresh one.
    #[test]
    fn density_and_age_do_what_they_say() {
        let covered = |density| sphere_points(3000).filter(|&p| sample(&spec(), 9, p, DATUM, density) != DATUM).count();
        let (sparse, dense) = (covered(0.2), covered(0.9));
        assert!(dense > 2 * sparse && sparse > 0, "{sparse} {dense}");
        let relief = |age| {
            sphere_points(3000).map(|p| (sample(&Spec { age, ..spec() }, 9, p, DATUM, 0.9) - DATUM).abs()).sum::<f32>()
        };
        assert!(relief(0.8) < 0.5 * relief(0.0));
    }

    #[test]
    fn negative_cells_hash_apart_from_positive_ones() {
        let a = cell_hash(1, 0, [-1, 0, 0]);
        let b = cell_hash(1, 0, [1, 0, 0]);
        assert_ne!(a, b);
        assert!(unit(u32::MAX) < 1.0);
    }
}

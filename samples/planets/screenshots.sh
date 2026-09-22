#!/usr/bin/env bash
set -euo pipefail
# Renders the sample set and contact sheets into $1. Run from the repo root.
# Contact sheets need ImageMagick (`magick`).
S="${1:?usage: samples/planets/screenshots.sh <out-dir>}"
mkdir -p "$S"
cargo run -q -p texture-graph-core --example planets >/dev/null
cargo build -q -p texture-graph-ui
B=target/debug/texture-graph
C=(--size 640 --background 05060a)
E=samples/planets/earthlike.tgraph; K=samples/planets/earthlike-clouds.tgraph; M=samples/planets/marslike.tgraph
for seed in 1 2 3; do
 for yaw in 0 130 250; do
  $B screenshot $E "$S/earth-s$seed-y$yaw.png" "${C[@]}" --seed $seed --yaw $yaw --tilt 23 --shell $K >/dev/null
  $B screenshot $M "$S/mars-s$seed-y$yaw.png" "${C[@]}" --seed $seed --yaw $yaw --tilt 25 >/dev/null
 done
done
$B screenshot $E "$S/earth-s1-bare.png" "${C[@]}" --seed 1 --yaw 60 >/dev/null
$B screenshot $E "$S/earth-s2-pole.png" "${C[@]}" --seed 2 --pitch 55 --shell $K >/dev/null
$B screenshot $M "$S/mars-s2-pole.png" "${C[@]}" --seed 2 --pitch 55 >/dev/null
$B screenshot $E "$S/earth-s4-iceage.png" "${C[@]}" --seed 4 --yaw 40 --tilt 23 --param ice=0.12 --param ocean=-0.03 --shell $K >/dev/null
$B screenshot $E "$S/earth-s5-dry.png" "${C[@]}" --seed 5 --yaw 40 --tilt 23 --param aridity=0.15 --param ocean=-0.04 --shell $K >/dev/null
$B screenshot $E "$S/earth-s6-ocean.png" "${C[@]}" --seed 6 --yaw 40 --tilt 23 --param ocean=0.06 --shell $K >/dev/null
$B screenshot $M "$S/mars-s4-dark.png" "${C[@]}" --seed 4 --yaw 40 --tilt 25 --param dark=0.08 --param dust=0.58,0.14,40 >/dev/null
cd "$S"
for p in earth mars; do
 for s in 1 2 3; do magick $p-s$s-y0.png $p-s$s-y130.png $p-s$s-y250.png +append row-$p-$s.png; done
 magick row-$p-1.png row-$p-2.png row-$p-3.png -append -resize 50% sheet-$p.png
done
magick earth-s1-bare.png earth-s2-pole.png mars-s2-pole.png mars-s4-dark.png +append a.png
magick earth-s4-iceage.png earth-s5-dry.png earth-s6-ocean.png +append b.png
magick a.png -resize 75% a.png
magick a.png b.png -append -resize 50% sheet-extra.png
rm a.png b.png row-*.png

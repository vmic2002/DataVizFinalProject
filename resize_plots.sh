#!/bin/bash

INPUT_DIR="odds_visualizations"
OUTPUT_DIR="small_plots"

mkdir -p "$OUTPUT_DIR"

for f in "$INPUT_DIR"/*.png; do
  magick "$f" \
    -resize 2000x \
    -density 150 \
    "$OUTPUT_DIR/$(basename "$f")"
done


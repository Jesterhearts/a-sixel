#!/usr/bin/env bash
set -euo pipefail

palette_size=""
dither=""
algorithm=""
show=0

valid_algorithms=(
    adu
    bit
    bit-merge-low
    bit-merge
    bit-merge-better
    bit-merge-best
    focal
    k-means
    k-medians
    median-cut
    octree
    wu
)

usage() {
    echo "Usage: $0 [-p palette_size] [-d no|sierra|sobol|bayer] [-a algorithm[,algorithm,...]] [-s]"
    echo "  -p  Palette size (default: 16 and 256)"
    echo "  -d  Dither mode: no, sierra, sobol, bayer"
    echo "  -a  Algorithm(s), comma-separated: ${valid_algorithms[*]}"
    echo "  -s  Show output"
    exit 1
}

validate_algorithm() {
    local alg="$1"
    for valid in "${valid_algorithms[@]}"; do
        [[ "$alg" == "$valid" ]] && return 0
    done
    echo "Invalid algorithm: $alg" >&2
    usage
}

while getopts "p:d:a:sh" opt; do
    case "$opt" in
        p) palette_size="$OPTARG" ;;
        d)
            case "$OPTARG" in
                no|sierra|sobol|bayer) dither="$OPTARG" ;;
                *) echo "Invalid dither: $OPTARG"; usage ;;
            esac
            ;;
        a) algorithm="$OPTARG" ;;
        s) show=1 ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [[ -n "$palette_size" ]]; then
    palette_sizes=("$palette_size")
else
    palette_sizes=(16 256)
fi

if [[ -n "$algorithm" ]]; then
    IFS=',' read -ra algorithms <<< "$algorithm"
    for alg in "${algorithms[@]}"; do
        validate_algorithm "$alg"
    done
else
    algorithms=("${valid_algorithms[@]}")
fi

test_images_path="test_images"
shopt -s nullglob
image_files=("$test_images_path"/*.png)
shopt -u nullglob

if [[ ${#image_files[@]} -eq 0 ]]; then
    echo "Error: No PNG files found in $test_images_path directory" >&2
    exit 1
fi

for ps in "${palette_sizes[@]}"; do
    for image_path in "${image_files[@]}"; do
        for alg in "${algorithms[@]}"; do
            cargo_args=(
                run --release
                --example image_viewer
                --features all-algorithms
                # --features dump-mse
                # --features dump-delta-e
                # --features dump-dssim
                # --features dump-phash
                --
                -i "$image_path"
                -f "$alg"
                -p "$ps"
            )

            if [[ -n "$dither" ]]; then
                cargo_args+=(-d "$dither")
            fi

            if [[ "$show" -ne 0 ]]; then
                cargo_args+=(-s)
            fi

            if ! cargo "${cargo_args[@]}"; then
                echo "Warning: Failed to process $(basename "$image_path") with $alg" >&2
            fi
        done
    done
done

param(
    [Parameter(Mandatory = $false)]
    [int]$PaletteSize = $null,

    [Parameter(Mandatory = $false)]
    [ValidateSet("no", "sierra", "sobol", "bayer")]
    [string]$Dither = $null,


    [Parameter(Mandatory = $false)]
    [ValidateSet("adu", "bit", "bit-merge-low", "bit-merge", "bit-merge-better", "bit-merge-best", "focal", "k-means", "k-medians", "median-cut", "octree", "wu")]
    [string]$Algorithm = $null,

    [Parameter(Mandatory = $false)]
    [int]$Show = $false
)

if ($PaletteSize) {
    $paletteSizes = @($PaletteSize)
}
else {
    $paletteSizes = @(16, 256)
}

if ($Algorithm) {
    $algorithms = @($Algorithm)
}
else {
    $algorithms = @("adu", "bit", "bit-merge-low", "bit-merge", "bit-merge-better", "bit-merge-best", "focal", "k-means", "k-medians", "median-cut", "octree", "wu")
}

# Get all images in test_images directory
$testImagesPath = "test_images"
$imageFiles = Get-ChildItem -Path $testImagesPath -Filter "*.png" | Select-Object -ExpandProperty Name

if ($imageFiles.Count -eq 0) {
    Write-Error "No PNG files found in $testImagesPath directory"
    exit 1
}

foreach ($paletteSize in $paletteSizes) {
    foreach ($algorithm in $algorithms) {
        foreach ($image in $imageFiles) {
            $imagePath = Join-Path $testImagesPath $image
        
            # Build the cargo command
            $cargoArgs = @(
                "run", "--release", 
                "--example", "image_viewer",
                "--features", "all-algorithms", 
                # "--features", "dump-mse", 
                # "--features", "dump-delta-e", 
                # "--features", "dump-dssim",
                # "--features", "dump-phash",
                "--",
                "-i", $imagePath,
                "-f", $algorithm,
                "-p", $paletteSize
            )
        
            # Add dither option if specified
            if ($Dither) {
                $cargoArgs += @("-d", $Dither)
            }

            if ($Show -ne 0) {
                $cargoArgs += "-s"
            }
        
            try {
                & cargo @cargoArgs
                if ($LASTEXITCODE -ne 0) {
                    Write-Warning "Failed to process $image with $algorithm (exit code: ${LASTEXITCODE})"
                }
            }
            catch {
                Write-Error "Error running cargo for $image with ${algorithm}: ${_}"
            }
        }
    }
}

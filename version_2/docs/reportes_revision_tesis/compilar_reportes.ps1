$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $ediciones = @('reporte_maestro_atractores_ocultos', 'reporte_maestro_lectura_estudio')
    foreach ($edicion in $ediciones) {
        $ErrorActionPreference = 'Continue'
        & latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build "$edicion.tex"
        $codigo = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        if ($codigo -ne 0) { throw "La compilación falló: $edicion" }
        Copy-Item -LiteralPath "build/$edicion.pdf" -Destination "$edicion.pdf" -Force
    }
} finally {
    Pop-Location
}

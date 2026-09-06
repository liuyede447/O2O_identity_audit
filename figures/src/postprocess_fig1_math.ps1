param(
    [Parameter(Mandatory=$true)][string]$InputPptx,
    [Parameter(Mandatory=$true)][string]$OutputPptx
)

$source = [System.IO.Path]::GetFullPath($InputPptx)
$target = [System.IO.Path]::GetFullPath($OutputPptx)
if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
    throw "Input PPTX not found: $source"
}
if ([System.IO.Path]::GetExtension($source) -ne '.pptx' -or [System.IO.Path]::GetExtension($target) -ne '.pptx') {
    throw 'Input and output must be PPTX files.'
}
Copy-Item -LiteralPath $source -Destination $target -Force

$app = New-Object -ComObject PowerPoint.Application
$deck = $null
try {
    $deck = $app.Presentations.Open($target, $false, $false, $false)
    $shape = $deck.Slides.Item(1).Shapes.Item('a-equivalent-formula')
    $range = $shape.TextFrame.TextRange
    $range.Text = "Eq.-area`rδ = κsg`rsg = √(wghg)"
    $range.Font.Name = 'Arial'
    $range.Font.Size = 7
    $range.Font.Bold = 0
    $range.Font.Italic = 0
    $range.ParagraphFormat.Alignment = 2

    # Format mathematical variables; every g is the subscript of the
    # immediately preceding s, w, or h. The first line contains no g.
    for ($index = 1; $index -le $range.Length; $index++) {
        $char = $range.Characters($index, 1)
        $value = $char.Text
        if ($value -in @('δ', 'κ', 's', 'w', 'h', 'g')) {
            $char.Font.Italic = -1
        }
        if ($value -eq 'g') {
            $char.Font.Subscript = -1
        }
    }
    $deck.Save()
}
finally {
    if ($null -ne $deck) { $deck.Close() }
    $app.Quit()
    [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($app) | Out-Null
}

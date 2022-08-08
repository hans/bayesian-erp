

// Questions
// - word annotations: SKIP1? SKIP2? GBG-LOOP?

baseDir = projectDir.parent
params.data_dir = "${baseDir}/data/gillis2021"
eeg_dir = file("${params.data_dir}/eeg")
textgrid_dir = file("${params.data_dir}/textgrids")

process convertTextgrid {
    input:
    file textgrid

    output:
    file "*.csv"

    script:
    outfile = textgrid.getName().replace(".TextGrid", ".csv")
    "python ${baseDir}/scripts/gillis2021/convert_textgrid.py ${textgrid} > ${outfile}"
}

/**
 * Match up force-aligned corpus with raw text corpus on a token-to-token level.
 * This will allow us to use token-level features computed on the latter corpus
 * together with the timing data from the former.
 */
process alignWithRawText {
    // TODO. For the moment we'll just use ground-truth surprisals.

    input:
    file textgrid
}


workflow {
    Channel.fromPath(textgrid_dir / "DKZ_*.TextGrid") | convertTextgrid
}


// Questions
// - word annotations: SKIP1? SKIP2? GBG-LOOP?

baseDir = projectDir.parent
params.data_dir = "${baseDir}/data/gillis2021"
eeg_dir = file("${params.data_dir}/eeg")
textgrid_dir = file("${params.data_dir}/textgrids")

params.model = "GroNLP/gpt2-small-dutch"
// Number of word candidates to consider in predictive model.
params.n_candidates = 10

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
 *
 * Outputs a tokenized version of the raw text and a CSV describing the alignment
 * between this tokenized version and the force-aligned corpus (FA herein).
 *
 * The output CSV is a many-to-many mapping between tokens in the force-aligned corpus
 * and the tokens in the raw text.
 *
 *   - textgrid_file: FA corpus for token
 *   - textgrid_idx: token index in FA corpus
 *   - tok_idx: token idx in tokenized text
 */
process alignWithRawText {
    // TODO. For the moment we'll just use ground-truth surprisals.

    input:
    path "*.csv" from force_aligned
    path raw_text

    output:
    path "tokenized.txt"
    path "aligned.csv"

    script:
    """
    python ${baseDir}/scripts/gillis2021/align_with_raw_text.py \
        -m ${params.model} \
        ${raw_text} \
        ${textgrids} \
        > aligned_text.csv
    """
}

/**
 * Produce an RRDataset from the aligned corpora for a single subject.
 */
process produceDataset {

    input:
    path tokenized_corpus
    path alignment_df
    path eeg_data

    output:
    path "dataset.pkl"

    script:
    """
    python ${baseDir}/scripts/gillis2021/produce_dataset.py \
        -m ${params.model} \
        -n ${params.n_candidates} \
        ${tokenized_corpus} \
        ${alignment_df} \
        ${eeg_data} \
        > dataset.pkl
    """

}

// /**
//  * Compute token-level predictive distributions.
//  *
//  * The output CSV contains the following columns:
//  *
//  *   - tok_idx: token idx
//  *   - candidate_idx: position in top-k predictive list. if 0, then this is the ground-truth
//  */
// process computePredictive {

//     input:
//     path "tokenized.txt"

//     output:
//     path "predictive.csv"

// }


workflow {
    // Collect data from all three force-aligned corpora.
    force_aligned_data = (Channel.fromPath(textgrid_dir / "DKZ_*.TextGrid") | convertTextgrid).collect()
}
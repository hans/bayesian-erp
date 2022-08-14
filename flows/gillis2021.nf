

// Questions
// - word annotations: SKIP1? SKIP2? GBG-LOOP?

baseDir = projectDir.parent
params.data_dir = "${baseDir}/data/gillis2021"
eeg_dir = file("${params.data_dir}/eeg")
textgrid_dir = file("${params.data_dir}/textgrids")
stim_dir = file("${params.data_dir}/predictors")
raw_text_dir = file("${params.data_dir}/raw_text")

params.model = "GroNLP/gpt2-small-dutch"
// Number of word candidates to consider in predictive model.
params.n_candidates = 10

params.eelbrain_env = "/home/jgauthie/.conda/envs/eeldev"
params.berp_env = "/home/jgauthie/.conda/envs/berp"

process convertTextgrid {
    input:
    path textgrid

    output:
    path "*.csv"

    script:
    outfile = textgrid.getName().replace(".TextGrid", ".csv")
    "python ${baseDir}/scripts/gillis2021/convert_textgrid.py ${textgrid} > ${outfile}"
}


/**
 * Convert Gillis' features from Eelbrain representation to numpy representation.
 */
process convertStimulusFeatures {
    conda params.eelbrain_env
    container null

    input:
    path stim_dir

    output:
    path "stimuli.npz"

    script:
    """
    python ${baseDir}/scripts/gillis2021/convert_features.py \
        ${stim_dir} stimuli.npz
    """
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
    input:
    path force_aligned_csvs
    path raw_text

    output:
    path "tokenized", emit: tokenized
    path "aligned_words.csv", emit: aligned_words
    path "aligned_phonemes.csv", emit: aligned_phonemes

    script:
    """
    python ${baseDir}/scripts/gillis2021/align_with_raw_text.py \
        -m ${params.model} \
        ${raw_text} \
        *.csv
    """
}


/**
 * Produce an RRDataset from the aligned corpora for a single subject.
 */
process produceDataset {

    container null
    conda params.berp_env

    input:
    path(tokenized_corpus_dir)
    path(aligned_words)
    path(aligned_phonemes)
    path eeg_data
    path stim_path

    output:
    path "*.pkl"

    script:
    """
    export PYTHONPATH=${baseDir}
    python ${baseDir}/scripts/gillis2021/produce_dataset.py \
        --model ${params.model} \
        --n_candidates ${params.n_candidates} \
        ${tokenized_corpus_dir} \
        ${aligned_words} ${aligned_phonemes} \
        ${eeg_data} ${stim_path}
    """

}


workflow {
    // Collect data from all three force-aligned corpora.
    force_aligned_data = convertTextgrid(Channel.fromPath(textgrid_dir / "DKZ_*.TextGrid")) \
        | collect

    // Prepare stimulus features.
    stimulus_features = convertStimulusFeatures(stim_dir)

    produceDataset(
        alignWithRawText(force_aligned_data, raw_text_dir),
        Channel.fromPath(eeg_dir),
        stimulus_features
    )
}
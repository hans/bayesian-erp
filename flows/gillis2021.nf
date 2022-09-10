

// Questions
// - word annotations: SKIP1? SKIP2? GBG-LOOP?

baseDir = projectDir.parent
params.data_dir = "${baseDir}/data/gillis2021"
eeg_dir = file("${params.data_dir}/eeg")
textgrid_dir = file("${params.data_dir}/textgrids")
stim_dir = file("${params.data_dir}/predictors")
raw_text_dir = file("${params.data_dir}/raw_text")
vocab_path = file("${params.data_dir}/vocab.pkl")
celex_path = file("${params.data_dir}/celex_dpw_cx.txt")

outDir = "${baseDir}/results/gillis2021"

params.model = "GroNLP/gpt2-small-dutch"
// Number of word candidates to consider in predictive model.
params.n_candidates = 10

params.eelbrain_env = "/home/jgauthie/.conda/envs/eeldev"
params.berp_env = "/home/jgauthie/.conda/envs/berp"

process convertTextgrid {
    input:
    path textgrid

    output:
    tuple val(story_name), path("${story_name}.csv")

    script:
    story_name = textgrid.baseName
    "python ${baseDir}/scripts/gillis2021/convert_textgrid.py ${textgrid} \
        > ${story_name}.csv"
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
        ${stim_dir} stimuli.npz \
        --drop_features "phoneme surprisal_0,phoneme entropy_0"
    """
}


/**
 * Match up force-aligned corpus with raw text corpus on a token-to-token level.
 * This will allow us to use token-level features computed on the latter corpus
 * together with the timing data from the former.
 *
 * Outputs a tokenized version of the raw text and a CSV describing the alignment
 * between this tokenized version and the force-aligned corpus.
 */
process alignWithRawText {
    container null
    conda params.berp_env
    tag "${story_name}"

    // DEV: don't support dkz_3 yet
    when:
    story_name != "DKZ_3"

    input:
    tuple val(story_name), path(force_aligned_csv), path(raw_text)

    output:
    tuple val(story_name), path("${story_name}.tokenized.txt"), \
        path("${story_name}.words.csv"), \
        path("${story_name}.phonemes.csv")

    script:
    """
    python ${baseDir}/scripts/gillis2021/align_with_raw_text.py \
        -m ${params.model} \
        ${raw_text} \
        ${force_aligned_csv}
    """
}


/**
 * Run a language model on the resulting aligned text inputs and generate
 * a NaturalLanguageStimulus, representing word- and phoneme-level prior
 * predictive distributions.
 */
process runLanguageModeling {
    container null
    conda params.berp_env
    tag "${story_name}"

    input:
    tuple val(story_name), path(tokenized), path(aligned_words), path(aligned_phonemes)

    output:
    tuple val(story_name), path("${story_name}.pkl")

    script:
    """
    export PYTHONPATH=${baseDir}
    python ${baseDir}/scripts/gillis2021/run_language_model.py \
        -m ${params.model} \
        -n ${params.n_candidates} \
        --vocab_path ${vocab_path} \
        --celex_path ${celex_path} \
        ${tokenized} \
        ${aligned_words} \
        ${aligned_phonemes}
    """
}


/**
 * Produce an RRDataset from the aligned corpora for a single subject.
 */
process produceDataset {

    container null
    conda params.berp_env

    publishDir "${outDir}/datasets"

    input:
    tuple val(story_name), path(eeg_data), \
        path(natural_language_stimulus), \
        path(tokenized), path(aligned_words), path(aligned_phonemes), \
        path(stim_features)

    output:
    path "*.pkl"

    script:
    """
    export PYTHONPATH=${baseDir}
    python ${baseDir}/scripts/gillis2021/produce_dataset.py \
        ${natural_language_stimulus} \
        ${aligned_words} ${aligned_phonemes} \
        ${eeg_data} ${stim_features}
    """

}


workflow {
    // Prepare stimulus features.
    stimulus_features = convertStimulusFeatures(stim_dir)

    raw_text = channel.fromPath(raw_text_dir / "*.txt") \
        | map { [it.baseName, it] }

    // Align with raw text representation.
    raw_textgrids = channel.fromPath(textgrid_dir / "DKZ_*.TextGrid")
    textgrids = raw_textgrids | convertTextgrid
    aligned = textgrids.join(raw_text) | alignWithRawText
    
    nl_stimuli = aligned | runLanguageModeling

    eeg_data = channel.fromPath(eeg_dir / "*" / "*.mat") | map {[it.parent.name, it]}
    // TODO only gets one per story?
    // Group by story and join with NL stimuli, then send to produceDataset.
    eeg_data.join(nl_stimuli).join(aligned).combine(stimulus_features) \
        // Analyze.
        // Produce dataset.
        | produceDataset
}

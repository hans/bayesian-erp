

baseDir = projectDir.parent
params.data_dir = "${baseDir}/data/gillis2021"
eeg_dir = file("${params.data_dir}/eeg")
textgrid_dir = file("${params.data_dir}/textgrids")
stim_dir = file("${params.data_dir}/predictors")
raw_text_dir = file("${params.data_dir}/raw_text")
vocab_path = file("${params.data_dir}/vocab.pkl")
celex_path = file("${params.data_dir}/celex_dpw_cx.txt")

confusion_args = "${params.data_dir}/confusion/phon2_conf_matrix_gate5.dat"

// Strip this from all EEG data files when computing subject names
EEG_SUFFIX = "_1_256_8_average_4_128"

/**
 * NB, resources on different phonological annotation schemes for dutch:
 *
 * CELEX: https://catalog.ldc.upenn.edu/docs/LDC96L14/dug_let.ps
 * CGN (Corpus Gesproken Nederlands) as annotated in Gillis et al. data:
 *   https://lands.let.ru.nl/cgn/doc_Dutch/topics/version_1.0/annot/phonetics/fon_prot.pdf
 */

outDir = "${baseDir}/results/gillis2021_avgpool"

params.model = "GroNLP/gpt2-small-dutch"
// Number of word candidates to consider in predictive model.
params.n_candidates = 20000

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
        ${stim_dir} stimuli.npz
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
 * Produce a BerpDataset from the aligned corpora for a single subject.
 */
process produceDataset {

    container null
    conda params.berp_env
    tag "${story_name}/${subject_name}"

    publishDir "${outDir}/datasets"

    input:
    tuple val(story_name), path(eeg_data), \
        path(natural_language_stimulus), \
        path(tokenized), path(aligned_words), path(aligned_phonemes), \
        path(stim_features)

    output:
    tuple val(subject_name), val(story_name), path("${story_name}.${subject_name}.pkl")

    script:
    subject_name = eeg_data.baseName.replace(EEG_SUFFIX, "")
    """
    export PYTHONPATH=${baseDir}
    python ${baseDir}/scripts/gillis2021/produce_dataset.py \
        ${natural_language_stimulus} \
        ${aligned_words} ${aligned_phonemes} \
        ${eeg_data} ${stim_features}

    """

}


process prepareConfusionMatrix {

    container null
    conda params.berp_env

    publishDir "${outDir}/confusion"

    input:
    path single_stim

    output:
    path "confusion.npz"

    script:
    """
    export PYTHONPATH=${baseDir}
    python ${baseDir}/scripts/gillis2021/prepare_confusion.py \
        ${confusion_args} \
        ${single_stim} \
        confusion.npz
    """

}


/**
 * Fit vanilla TRF encoders and learn alphas per-subject.
 */
process fitVanillaEncoders {
    container null
    conda params.berp_env
    tag "${subject_name}"

    publishDir "${outDir}/models_vanilla"

    input:
    tuple val(subject_name), val(datasets)

    output:
    path("${subject_name}")

    script:
    dataset_path_str = datasets.collect { it[1] }.join(",")
    """
    export PYTHONPATH=${baseDir}
    python ${baseDir}/scripts/fit_em.py \
        model=trf \
        cv=search_alpha \
        'dataset.paths=[${dataset_path_str}]' \
        hydra.run.dir="${subject_name}" \
        "dataset.drop_X_variable=[recognition_onset]"
    """
}


/**
 * Fit Berp TRF encoder using pretrained vanilla encoders for pipeline init,
 * using expectation maximization.
 */
process fitBerp {
    container null
    conda params.berp_env

    publishDir "${outDir}/models_berp"

    input:
    path datasets
    path vanilla_models
    path confusion

    output:
    path "berp"

    script:
    vanilla_pipelines = (vanilla_models.collect { it + "/params/pipeline.pkl" }.join(","))
    dataset_path_str = datasets.join(",")
    """
    export PYTHONPATH=${baseDir}
    python ${baseDir}/scripts/fit_em.py \
        model=trf-em \
        'dataset.paths=[${dataset_path_str}]' \
        'model.pretrained_pipeline_paths=[${vanilla_pipelines}]' \
        model.confusion_path=${confusion} \
        cv=off \
        solver=adam_em \
        hydra.run.dir="berp"
    """
}


/**
 * Fit Berp TRF encoder using a grid search.
 */
process fitBerpGrid {
    container null
    conda params.berp_env

    publishDir "${outDir}/models_berp_grid"

    input:
    path datasets
    path stimuli
    path confusion

    output:
    path "berp-fixed"

    script:
    dataset_path_str = datasets.join(",")

    // Produce stimulus lookup dict
    stimulus_path_str = stimuli.collect { "${it.baseName}:'${it}'" }.join(",")
    stimulus_path_str = "\"{${stimulus_path_str}}\""
    """
    export PYTHONPATH=${baseDir}
    python ${baseDir}/scripts/fit_em.py \
        model=trf-berp-fixed \
        'dataset.paths=[${dataset_path_str}]' \
        +dataset.stimulus_paths=${stimulus_path_str} \
        model.confusion_path=${confusion} \
        cv=search_alpha_threshold_lambda \
        solver=adam \
        hydra.run.dir="berp-fixed"
    """
}


/**
 * Fit unitary TRF encoder and individual TRFs per subject, for
 * direct comparability with fitBerpGrid.
 */
process fitUnitaryVanillaEncoder {
    container null
    conda params.berp_env

    publishDir "${outDir}/models_vanilla_unitary"

    input:
    path datasets
    path stimuli

    output:
    path("berp-vanilla-unitary")

    script:
    dataset_path_str = datasets.join(",")

    // Produce stimulus lookup dict
    stimulus_path_str = stimuli.collect { "${it.baseName}:'${it}'" }.join(",")
    stimulus_path_str = "\"{${stimulus_path_str}}\""
    """
    export PYTHONPATH=${baseDir}
    python ${baseDir}/scripts/fit_em.py \
        model=trf \
        cv=search_alpha \
        'dataset.paths=[${dataset_path_str}]' \
        +dataset.stimulus_paths=${stimulus_path_str} \
        hydra.run.dir="berp-vanilla-unitary" \
        "dataset.drop_X_variable=[recognition_onset]"
    """
}


process produceAverageDataset {
    container null
    conda params.berp_env

    publishDir "${outDir}"
    tag "${story_name}"

    input:
    tuple val(story_name), path(datasets)

    output:
    tuple val(story_name), path("${story_name}.average.pkl")

    script:
    dataset_path_str = datasets.join(" ")
    """
    export PYTHONPATH=${baseDir}
    python ${baseDir}/scripts/gillis2021/produce_average_dataset.py \
        -o ${story_name}.average.pkl \
        -n ${story_name}/average \
        ${dataset_path_str}
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

    // EEG recordings, grouped by story
    eeg_data = channel.fromPath(eeg_dir / "*" / "*.mat") | map {[it.parent.name, it]}
    // Group by story and join with NL stimuli, then send to produceDataset.
    full_datasets = eeg_data.combine(nl_stimuli, by: 0).combine(aligned, by: 0).combine(stimulus_features) \
        // Produce dataset.
        | produceDataset

    // Prepare confusion matrix data with a sample dataset (any is good)
    confusion = prepareConfusionMatrix(nl_stimuli.map { it[1] } | first)

    ///////

    // TODO create a separate workflow for multi-subject fit

    // // Group by subject and fit vanilla encoders.
    // vanilla_results = full_datasets | map { tuple(it[0], tuple(it[1], it[2])) } | groupTuple() \
    //      | fitVanillaEncoders

    // // Fit unitary vanilla encoder (single alpha).
    // vanilla_unitary = fitUnitaryVanillaEncoder(full_datasets.collect { it[2] })

    // // fitBerp(full_datasets.collect { it[2] }, vanilla_results.collect(), confusion)
    // fitBerpGrid(full_datasets.collect { it[2] }, confusion)

    ////////

    // Produce one average dataset per story.
    full_datasets_by_story = full_datasets.map { [it[1], it[2]] }.groupTuple()
    average_dataset = produceAverageDataset(full_datasets_by_story)

    // vanilla = fitUnitaryVanillaEncoder(
    //     average_dataset.collect { it[1] },
    //     nl_stimuli.collect { it[1] })

    // Fit Berp model on average dataset.
    fitBerpGrid(
        average_dataset.collect { it[1] },
        nl_stimuli.collect { it[1] },
        confusion)
}

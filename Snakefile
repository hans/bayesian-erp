from snakemake.utils import min_version
min_version("6.0")

configfile: "config.yaml"


def make_merged_config(dataset_name, config=config):
    """
    Merge global pipeline config with dataset-specific config.
    """

    ret = config | config["datasets"][dataset_name]
    ret.pop("datasets")
    return ret


module gillis2021:
    snakefile: "workflow/gillis2021/Snakefile"
    config: make_merged_config("gillis2021")
    prefix: "workflow/gillis2021"

use rule * from gillis2021 as gillis2021_*


module heilbron2022:
    snakefile: "workflow/heilbron2022/Snakefile"
    config: make_merged_config("heilbron2022")
    prefix: "workflow/heilbron2022"

use rule * from heilbron2022 as heilbron2022_*
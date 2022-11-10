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
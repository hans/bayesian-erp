from snakemake.utils import min_version
min_version("6.0")

configfile: "config.yaml"

module gillis2021:
    snakefile: "workflow/gillis2021/Snakefile"
    config: config["datasets"]["gillis2021"] | config
    prefix: "workflow/gillis2021"

use rule * from gillis2021 as gillis2021_*
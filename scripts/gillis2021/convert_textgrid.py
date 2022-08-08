import sys

import pandas as pd
import textgrid


def textgrid_tier_to_csv(tg, tier_name):
    tier = tg.tiers[tg.getNames().index(tier_name)]
    df = pd.DataFrame(columns=["start", "end", "text"])
    for interval in tier.intervals:
        df.loc[len(df)] = [interval.minTime, interval.maxTime, interval.mark]
    
    return df


def main():
    if len(sys.argv) != 2:
        print("Usage: {} <textgrid_file>".format(sys.argv[0]))
        sys.exit(1)

    tg = textgrid.TextGrid.fromFile(sys.argv[1])

    source_tier_names = ["RT_WordSegments", "RT_PhoneSegments"]
    target_tier_names = ["words", "phonemes"]

    dfs = [textgrid_tier_to_csv(tg, tier_name) for tier_name in source_tier_names]
    df = pd.concat(dfs, axis=0, names=["tier"], keys=target_tier_names)

    df.to_csv(sys.stdout)


if __name__ == "__main__":
    main()
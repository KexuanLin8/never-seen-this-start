"""Create a new Evo-1 fine-tuning variant from existing ones (dataset config + training command).

  python make_variant.py --name startmid --from start mid      # union of both variants' datasets
  python make_variant.py --name starthalf --from start --replace synth_lerobot/start_=synth_lerobot/starthalf_
  python make_variant.py --name orig_s2 --from orig            # repeat run (train.py is not seeded)

Writes $W/ft/evo1_ft_<name>/{dataset_config.yaml, train_cmd.sh}; every dataset already carries the released
normalisation statistics in meta/stats.json (prepare_evo1_finetune.py). Run in the Evo1 env.
"""

import argparse
from pathlib import Path

import yaml

W = Path("$W")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--from", dest="src", nargs="+", required=True, help="existing variants to merge")
    p.add_argument("--replace", action="append", default=[], help="old=new substitution in dataset paths")
    a = p.parse_args()

    out = W / "ft" / f"evo1_ft_{a.name}"
    if (out / "train_cmd.sh").exists():
        raise SystemExit(f"{out} already exists")
    cfg, groups = None, {}
    for v in a.src:
        c = yaml.safe_load((W / "ft" / f"evo1_ft_{v}" / "dataset_config.yaml").read_text())
        cfg = cfg or c
        (arm, g), = c["data_groups"].items()
        for name, spec in g.items():
            if name in groups and groups[name]["path"] != spec["path"]:
                # e.g. FT-chain was made from FT-start, so both call their synthetic sets synth_start_<suite>
                name = f"{name}__{Path(spec['path']).name}"
            groups[name] = spec
    for name, g in groups.items():
        for rep in a.replace:
            old, new = rep.split("=", 1)
            g["path"] = g["path"].replace(old, new)
        if not (Path(g["path"]) / "meta" / "stats.json").exists():
            raise SystemExit(f"{name}: {g['path']} has no meta/stats.json")
    cfg["data_groups"] = {arm: groups}
    cfg["datasets_manifest"] = f"manifest_evo1_ft_{a.name}.pkl"
    out.mkdir(parents=True)
    (out / "dataset_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    cmd = (W / "ft" / f"evo1_ft_{a.src[0]}" / "train_cmd.sh").read_text()
    cmd = cmd.replace(f"evo1_ft_{a.src[0]}", f"evo1_ft_{a.name}")
    (out / "train_cmd.sh").write_text(cmd)
    print(f"{out}: {len(groups)} datasets")
    for name, g in groups.items():
        print(f"  {name}: {g['path']}")


if __name__ == "__main__":
    main()

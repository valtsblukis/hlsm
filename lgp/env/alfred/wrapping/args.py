from lgp.env.alfred.wrapping.paths import get_alfred_root_path


def get_faux_args():
    alfred_root_path = get_alfred_root_path()
    args = type("FauxArgs",
                (object,),
                {
                    "reward_config": f"{alfred_root_path}/models/config/rewards.json"
                })()
    return args
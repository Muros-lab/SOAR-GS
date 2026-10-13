from .object_confidence import ObjectConfidencePlanner


def get_planner(cfg, device):
    planner_cfg = cfg.planner
    if planner_cfg.type == "object_confidence":
        return ObjectConfidencePlanner(planner_cfg, device)
    else:
        raise NotImplementedError(f"Unknown planner type: {planner_cfg.type}")

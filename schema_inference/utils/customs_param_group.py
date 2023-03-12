import re
import logging
from typing import Iterable, List, Dict, Any, Tuple
import collections

import torch.nn as nn

from cv_lib.utils import to_json_str


def customs_param_group(
    named_parameters: Iterable[Tuple[str, nn.Parameter]],
    groups: List[Dict[str, Any]],
    drop_remain: bool = False
) -> List[Dict[str, nn.Parameter]]:
    """
    Args:
        groups: list of group configuration, for example
            {
                pattern: str,
                cfg: Dict[str, Any]
            }
    """
    logger = logging.getLogger("cumtoms_param_group")

    def select(pattern: str):
        out_dict = collections.OrderedDict()
        for name in sorted(all_parameters):
            if re.match(pattern, name):
                out_dict[name] = all_parameters.pop(name)
        return out_dict

    all_parameters = dict(named_parameters)
    params = list()
    for group in groups:
        pattern = group["pattern"]
        cfg = group.get("cfg", dict())
        param_dict = select(pattern)
        num_params = len(param_dict)
        assert num_params > 0, "no matched for pattern {}".format(pattern)
        params.append(dict(params=param_dict.values(), **cfg))
        logger.info(
            "Added %d parameters to group `%s`, with config:\n%s",
            num_params, pattern, to_json_str(cfg)
        )

    if not drop_remain:
        if len(all_parameters) == 0:
            logger.warning("No parameters are added to default group")
        else:
            params.append(dict(params=all_parameters.values()))
            logger.info("Added %d parameters to default group", num_params)
    else:
        for param in all_parameters.values():
            param.requires_grad_(False)
        logger.info(
            "All others parameters (%d) are set to no grad: %s",
            len(all_parameters),
            str(list(all_parameters.keys()))
        )

    return params


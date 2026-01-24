# Reward Models for RLHF
#
# Original implementation by @myhott163com
# Source: https://github.com/myhott163com/RLHF_ORM_PRM
# License: MIT
#
# Adapted for RLHF Book (https://rlhfbook.com) by Nathan Lambert

from .train_orm import OutcomeRewardModel
from .train_prm import ProcessRewardModel


__all__ = ["OutcomeRewardModel", "ProcessRewardModel"]

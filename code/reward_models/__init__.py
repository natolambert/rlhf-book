# Reward Models for RLHF
#
# ORM/PRM: Original implementation by @myhott163com
# Source: https://github.com/myhott163com/RLHF_ORM_PRM
# License: MIT
#
# Preference RM: Adapted for RLHF Book by Nathan Lambert
#
# See Chapter 7 of RLHF Book (https://rlhfbook.com) for theoretical background.

from .train_orm import OutcomeRewardModel
from .train_preference_rm import PreferenceRewardModel
from .train_prm import ProcessRewardModel


__all__ = ["OutcomeRewardModel", "ProcessRewardModel", "PreferenceRewardModel"]

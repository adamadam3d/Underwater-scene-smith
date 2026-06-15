"""Accent agents for anchoring secondary marine structures to the reef.

The accent stage attaches decorative/secondary lifeforms to the primary reef
structures produced by the Reef Agent. It is split into two sub-agents that
share the same placement machinery but target different surface families:

- ``wall``: reef-surface accents on vertical/steep faces (e.g. sea fans /
  gorgonians, tube sponges) oriented to the simulated water current.
- ``ceiling``: overhead / canopy accents anchored to overhangs and the
  undersides of arches and ledges (e.g. hanging sponges, soft corals).
"""

from reefsmith.accent_agents.wall.base_wall_agent import BaseWallAgent
from reefsmith.accent_agents.wall.stateful_wall_agent import StatefulWallAgent
from reefsmith.accent_agents.ceiling.base_ceiling_agent import BaseCeilingAgent
from reefsmith.accent_agents.ceiling.stateful_ceiling_agent import StatefulCeilingAgent

__all__ = [
    "BaseWallAgent",
    "StatefulWallAgent",
    "BaseCeilingAgent",
    "StatefulCeilingAgent",
]

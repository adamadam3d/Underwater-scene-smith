"""Stateful Accent Agent entry point (Stage 3 of the ReefSmith pipeline).

The Accent Agent anchors secondary structural/decorative marine life (sea fans /
gorgonians, tube sponges, soft corals) to the primary reef structures produced
by the Reef Agent, aligning them with the simulated water current.

Implementation note: the accent stage is realized by two cooperating sub-agents
that share the same placement machinery but target different surface families:

- :class:`StatefulWallAccentAgent` - reef-surface accents on vertical/steep
  faces (the primary "accent" role; aliased here as ``StatefulAccentAgent``).
- :class:`StatefulCanopyAccentAgent` - overhead / canopy accents anchored to
  overhangs and the undersides of arches and ledges.

These retain ``Wall``/``Ceiling`` in their underlying class names for
serialization and checkpoint compatibility with the shared scene data model.
"""

from reefsmith.accent_agents.wall.stateful_wall_agent import (
    StatefulWallAgent as StatefulWallAccentAgent,
)
from reefsmith.accent_agents.ceiling.stateful_ceiling_agent import (
    StatefulCeilingAgent as StatefulCanopyAccentAgent,
)

# Primary accent agent (reef-surface accents) under the architecture's name.
StatefulAccentAgent = StatefulWallAccentAgent

__all__ = [
    "StatefulAccentAgent",
    "StatefulWallAccentAgent",
    "StatefulCanopyAccentAgent",
]

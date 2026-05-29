# ReefSmith Architecture (Underwater Adaptation)

## 1. High-Level Overview
**ReefSmith** is an agentic system for generating simulation-ready underwater coral reef environments and marine ecosystems from natural language text prompts. Adapted from the indoor-focused SceneSmith architecture, ReefSmith utilizes a sequential, multi-stage pipeline where specialized LLM agents cooperate to design seabed topographies, place large structural corals and rocks, and add intricate micro-habitats. 

The project integrates with 3D generative AI backends (including SAM3D and Hunyuan3D), Drake for rigid body and fluid dynamics simulation, Blender for underwater rendering (caustics, volumetric scattering), and Flask/Hydra for service orchestration. The ultimate goal of ReefSmith is to produce rich, physically valid underwater environments ready for downstream marine robotics simulation (e.g., ROV/AUV navigation, grasping) and ecological modeling tasks.

## 2. The Generative Ecological Pipeline
The core of ReefSmith is a multi-stage generation pipeline (`configurations/experiment/underwater_scene_generation.yaml`) that orchestrates a series of specialized ecological agents in sequence. Each agent acts upon the output of the previous one, building the environment from the seafloor up.

### Stage 1: Seabed Topography Design
The **Seabed Agent** (`reefsmith/seabed_agents/stateful_seabed_agent.py`) designs the geological skeleton of the scene.
- **Inputs:** User prompt, desired depth profile, bounding box.
- **Logic:** It procedurally generates a 3D topographical mesh representing the seafloor. It manages the placement of sand patches, exposed rock foundations, trenches, and drop-offs.
- **Constraints:** Ensures continuous nav-mesh viability for bottom-crawling robots and creates logical structural anchors for future coral placement.

### Stage 2: Structural Reef Placement
The **Reef Agent** (`reefsmith/reef_agents/stateful_reef_agent.py`) places primary structural objects.
- **Inputs:** Seabed topography map, flow channel vectors.
- **Logic:** Identifies functional foundation zones to place large-scale structures (e.g., massive brain corals, boulder clusters, staghorn thickets). 
- **Constraints:** Balances organic clustering aesthetics with physical constraints like ROV clearance corridors (ensuring an AUV can pass through) and simulated water flow channels.

### Stage 3: Coral & Sponge Accents
The **Accent Agent** (`reefsmith/accent_agents/stateful_accent_agent.py`) adds secondary structural and decorative items.
- **Inputs:** Base reef structure, current direction metadata.
- **Logic:** Manages the placement of items like large tube sponges, sea fans (gorgonians), and kelp.
- **Constraints:** Ensures objects are naturally anchored to the primary reef structures and geometrically aligned with simulated water currents (e.g., sea fans facing the current).

### Stage 4: Micro-Habitat Arrangement
The **Micro-Habitat Agent** (`reefsmith/micro_habitat_agents/stateful_micro_habitat_agent.py`) focuses on small, interactive, or detailed lifeforms.
- **Inputs:** High-resolution mesh of the reef structures, micro-scale clearance data.
- **Logic:** Identifies crevices, overhangs, and support surfaces on the reef and arranges items like starfish, anemones, sea urchins, and crabs.
- **Constraints:** Objects must be securely anchored or resting in valid ecological niches (not floating), making the ecosystem feel "alive" and ready for fine-motor interaction by ROV manipulators.

### Generative 3D Marine Asset Creation
For marine objects not present in existing HSSD or Objaverse libraries, the **Geometry Generation Server** (`reefsmith/agent_utils/geometry_generation_server/geometry_generation.py`) kicks in.
- **Process:** An LLM generates a highly specific descriptive prompt (e.g., "Macro photography of a purple branching sponge, isolated on a black background, no water distortion"). 
- **Generation:** Image-to-3D backends (like SAM3D) generate custom 3D meshes of these specific corals or rocks.
- **Refinement:** The VLM (Vision-Language Model) reviews the output to ensure it matches the prompt and scales it correctly before passing it to the simulation engine.

## 3. Core Architectural Layers
The codebase is structured into 10 fundamental layers, adapted for marine environment generation:

1. **Infrastructure & Configuration**
   - *Description:* Global configuration files, Docker specifications, and project-level metadata (e.g., `underwater_scene_generation.yaml`).
2. **Low-Level Utilities**
   - *Description:* Generic helper functions for networking, paths, package management, and basic math (quaternions, coordinate transforms).
   - *Examples:* `reefsmith/utils/`
3. **Core Agent Framework**
   - *Description:* Base classes, logging, checkpointing, state management, and common agent utilities (like LLM/VLM interfacing) used by all specialized agents.
   - *Examples:* `reefsmith/agent_utils/base_stateful_agent.py`
4. **Simulation & Geometry Engine**
   - *Description:* Core logic for interfacing with Drake physics. For ReefSmith, this includes standard SDF generation for collision/mass, plus approximations for buoyancy, drag coefficients, and hydrodynamic forces.
   - *Examples:* `reefsmith/agent_utils/sdf_generator.py`, `reefsmith/agent_utils/blender/`
5. **Specialized Retrieval Services**
   - *Description:* Flask/FastAPI servers and clients for retrieving 3D marine assets, underwater textures (e.g., sand, porous rock), and articulated structures.
   - *Examples:* `reefsmith/agent_utils/hssd_retrieval_server/`
6. **Ecological Agents**
   - *Description:* The specialized domain agents that execute the pipeline stages (Seabed, Reef, Accent, Micro-Habitat).
   - *Examples:* `reefsmith/seabed_agents/`, `reefsmith/reef_agents/`
7. **ROV/AUV Evaluation**
   - *Description:* Tools and agents for evaluating marine robot policies. Validates navigation, buoyancy control, and task success (e.g., sample collection) in generated reefs.
   - *Examples:* `reefsmith/robot_eval/`
8. **Prompts & Metadata**
   - *Description:* Agent prompt templates tailored for underwater generation (YAML files guiding LLMs on marine ecology, spacing, and image generation constraints).
   - *Examples:* `reefsmith/prompts/`
9. **Scripts & Entry Points**
   - *Description:* CLI tools, batch processing scripts, and main application entry points.
   - *Examples:* `scripts/export_scene_to_mujoco.py`, `main.py`
10. **Testing Suite**
    - *Description:* Unit and integration tests covering the entire underwater pipeline, including mocks for marine assets and fluid dynamics edge cases.
    - *Examples:* `tests/`

## 4. Codebase Navigation (Guided Tour)

1. **Welcome to ReefSmith (`README.md`)**
   - Start here for the project overview, installation instructions, and basic usage examples for generating a coral reef from a text prompt.
2. **The Generation Pipeline (`configurations/experiment/underwater_scene_generation.yaml`)**
   - Review how the system orchestrates the series of ecological agents. This configuration defines the sequence, hyperparameters, and endpoints for the generative servers.
3. **Stage 1: Seabed Topography (`reefsmith/seabed_agents/stateful_seabed_agent.py`)**
   - Dive into the code where the Seabed Agent procedurally turns abstract depth requirements into a 3D topographical mesh (sand, rock, trenches) using SDF logic.
4. **Stage 2: Structural Reef Placement (`reefsmith/reef_agents/stateful_reef_agent.py`)**
   - See how the Reef Agent parses the topographical map, identifies foundation zones, and places primary corals and boulders while balancing organic grouping with ROV volumetric clearance constraints.
5. **Stage 3: Coral & Sponge Accents (`reefsmith/accent_agents/stateful_accent_agent.py`)**
   - Explore how the Accent Agent manages secondary items like large sponges and gorgonians, using raycasting to ensure they are logically anchored to surfaces and aligned with simulated water currents.
6. **Stage 4: Micro-Habitat Arrangement (`reefsmith/micro_habitat_agents/stateful_micro_habitat_agent.py`)**
   - Look at the fine-grained placement logic where this agent identifies support crevices on rocks/corals and arranges small marine life to make the reef feel ecologically complete.
7. **Generative 3D Marine Asset Creation (`reefsmith/agent_utils/geometry_generation_server/geometry_generation.py`)**
   - Examine the core innovation: on-demand marine asset generation. This module manages the API calls to Vision-Language Models (VLM) and image-to-3D backends to generate custom 3D meshes of specific corals or rocks.
8. **Simulation Readiness & Hydrodynamics (`reefsmith/agent_utils/sdf_generator.py`)**
   - Discover how generated meshes are made simulation-ready. This module generates Drake-compatible SDF files, wrapping meshes in collision volumes and assigning physical properties (mass, inertia, and estimated buoyancy/drag) suitable for underwater physics simulation.
9. **Robot Task Evaluation (`reefsmith/robot_eval/success_validation/validator_agent.py`)**
   - Finally, review the evaluation loop. The Validator Agent uses visual and geometric checks to confirm if an underwater robot (ROV/AUV) can successfully navigate the reef or perform tasks (e.g., 'inspect brain coral') without colliding with delicate structures.
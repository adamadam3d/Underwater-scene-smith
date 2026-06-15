# ReefSmith: System Transformation Prompt

You are an expert software engineer and systems architect. Your task is to refactor the **SceneSmith** (indoor scene generation) codebase into **ReefSmith**, a system dedicated to generating simulation-ready underwater coral reefs and marine environments.

The project relies on a multi-stage pipeline of LLM agents orchestrating 3D generative AI (like SAM3D/Hunyuan3D), Drake for physics simulation, and Blender for rendering.

You must use the `ARCHITECTURE_UNDERWATER.md` file located in the project root as your ultimate source of truth for the target state.

## Phase 1: Core Domain Translation
The first step is to rename and refactor the core semantic domains throughout the codebase. You will need to carefully rename directories, files, base classes, and their corresponding references. 

Perform the following domain shifts:
1. **Floor Plan Agent $\rightarrow$ Seabed Agent:**
   - Transform `reefsmith/seabed_agents/` to `reefsmith/seabed_agents/`.
   - Update `stateful_floor_plan_agent.py` to `stateful_seabed_agent.py`.
   - Change logic from generating walls/rooms/doors to generating topographical boundaries, sand patches, rock foundations, trenches, and drop-offs.
2. **Furniture Agent $\rightarrow$ Reef Agent:**
   - Transform `reefsmith/reef_agents/` to `reefsmith/reef_agents/`.
   - Change logic from placing tables and chairs to placing structural marine objects (e.g., massive brain corals, boulder clusters, staghorn thickets) while respecting water flow channels and ROV clearance.
3. **Wall/Ceiling Agents $\rightarrow$ Accent Agents:**
   - Transform `reefsmith/accent_agents/wall/` and `reefsmith/accent_agents/ceiling/` to `reefsmith/accent_agents/`.
   - Change logic from hanging art and lights to anchoring secondary decorative items (like sea fans or tube sponges) to the primary reef structures.
4. **Manipuland Agent $\rightarrow$ Micro-Habitat Agent:**
   - Transform `reefsmith/micro_habitat_agents/` to `reefsmith/micro_habitat_agents/`.
   - Change logic from placing small objects on tables to nestling micro-life (starfish, anemones, urchins) into crevices and on the surfaces of rocks and corals.

## Phase 2: Generative Prompts & VLM Editing
The system relies on YAML files in `reefsmith/prompts/` to instruct the LLMs and VLMs. Update these to reflect the underwater domain.

1. **Asset Generation (`asset_image_initial.yaml`):**
   - Ensure the prompt strictly forbids water distortion, caustics, floating particles, and shadows. It must request a clean, macro-photography-style image against a solid opaque background (e.g., flat black) suitable for 3D mesh conversion.
2. **Context Images for Placement:**
   - Replace top-down "empty room" context prompts with "empty sandy seabed or rocky foundation" prompts.
   - Instruct the VLM to anchor objects realistically (no floating) and respect natural trenches instead of architectural doors/windows.

## Phase 3: Configuration & Pipeline Orchestration
1. **Hydra Configurations:** Update the pipeline orchestration in `configurations/experiment/`. Rename `underwater_scene_generation.yaml` to `underwater_scene_generation.yaml` and update the agent sequence to: `Seabed -> Reef -> Accent -> Micro-Habitat`.
2. **Robot Evaluation:** Update `reefsmith/robot_eval/` (which becomes `reefsmith/robot_eval/`) to validate marine robot (ROV/AUV) policies instead of terrestrial manipulation, changing predicates from things like "place item on table" to "inspect coral structure".

## Constraints and Requirements
- Do not blindly search-and-replace strings. You must read the implementation of the agents and tools to ensure the *logic* makes sense for an underwater environment (e.g., 2D grid pathfinding for a human in a room might need to become 3D volumetric clearance checking for an ROV).
- Maintain all existing integrations with Drake (SDF generation) and Blender (rendering).
- Maintain the current architecture's reliance on `stateful_*_agent.py` subclasses inheriting from the core agent framework.
- Thoroughly test the new pipeline using the provided testing suites in `tests/` (which will also need their mocks and fixtures updated to marine objects).

Begin by reading the `ARCHITECTURE_UNDERWATER.md` file, then create a step-by-step plan for the refactor before modifying any code.
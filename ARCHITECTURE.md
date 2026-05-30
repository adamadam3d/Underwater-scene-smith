# SceneSmith Architecture

## 1. High-Level Overview
**SceneSmith** is an agentic system for generating simulation-ready indoor scenes from text prompts. It utilizes a multi-stage pipeline where specialized LLM agents cooperate to design full house layouts, place furniture, and add small-scale details. The project integrates with 3D generative AI backends (including Drake, Blender, Flask, and Hydra) to produce rich, physically valid environments ready for downstream robotics simulation and tasks.

## 2. The Generation Pipeline
The core of SceneSmith is a multi-stage generation pipeline that orchestrates a series of specialized agents in sequence. The pipeline configuration defines the sequence and endpoints for the generative servers that power 3D asset creation:

- **Stage 1: Floor Plan Design:** The Floor Plan Agent designs the architectural skeleton of the scene, managing room placement, wall geometry, and structural openings.
- **Stage 2: Furniture Placement:** The Furniture Agent identifies functional zones and places primary objects (e.g., tables, chairs). It balances aesthetic arrangements with physical constraints like reachability and clearance.
- **Stage 3: Wall & Ceiling Accents:** Wall and Ceiling Agents add decorative and functional secondary items, ensuring lighting and artwork are logically aligned with the layout.
- **Stage 4: Manipuland Arrangement:** The Manipuland Agent focuses on small, interactive objects (manipulands). It identifies support surfaces on furniture and arranges items to make the scene "lived-in" and ready for robot interaction.
- **Generative 3D Asset Creation:** For objects not in existing libraries, on-demand generative backends use image-to-3D models to create custom 3D meshes, which are then processed for physical validity.

## 3. Core Architectural Layers
The codebase is structured into 10 fundamental layers, designed for modularity and scalability:

1. **Infrastructure & Configuration**
   - *Description:* Global configuration files, Docker specifications, and project-level metadata.
2. **Low-Level Utilities**
   - *Description:* Generic helper functions for networking, paths, package management, and basic math.
   - *Examples:* `scenesmith/utils/`
3. **Core Agent Framework**
   - *Description:* Base classes, logging, checkpointing, and common agent utilities used by all specialized agents.
   - *Examples:* `scenesmith/agent_utils/`
4. **Simulation & Geometry Engine**
   - *Description:* Core logic for interfacing with Drake physics, SDF generation, and mesh processing.
   - *Examples:* `scenesmith/agent_utils/geometry_generation_server/`, `scenesmith/agent_utils/blender/`
5. **Specialized Retrieval Services**
   - *Description:* Flask/FastAPI servers and clients for asset, material, and articulated object retrieval.
   - *Examples:* `scenesmith/agent_utils/*_retrieval_server/`
6. **Architectural Agents**
   - *Description:* Specialized agents for floor plan design, wall management, and furniture placement.
   - *Examples:* `scenesmith/floor_plan_agents/`, `scenesmith/furniture_agents/`, `scenesmith/wall_agents/`, `scenesmith/manipuland_agents/`
7. **Robot Evaluation**
   - *Description:* Tools and agents for evaluating robot policies and validating task success in generated scenes.
   - *Examples:* `scenesmith/robot_eval/`
8. **Prompts & Metadata**
   - *Description:* Agent prompt templates and registry systems.
   - *Examples:* `scenesmith/prompts/`
9. **Scripts & Entry Points**
   - *Description:* CLI tools, batch processing scripts, and main application entry points.
   - *Examples:* `scripts/`, `main.py`
10. **Testing Suite**
    - *Description:* Unit and integration tests covering the entire pipeline.
    - *Examples:* `tests/`

## 4. Codebase Navigation
This guided tour helps new developers navigate the critical components of the SceneSmith project sequentially:

- **Welcome to SceneSmith** (`README.md`): SceneSmith is an agentic system for generating simulation-ready indoor scenes from text prompts. This tour follows the multi-stage pipeline where specialized agents cooperate to design full house layouts, place furniture, and add small-scale details.
- **The Generation Pipeline** (`configurations/experiment/indoor_scene_generation.yaml`): The system orchestrates a series of agents (floor plan, furniture, wall, ceiling, and manipuland). This configuration defines the sequence and endpoints for the generative servers that power the 3D asset creation.
- **Stage 1: Floor Plan Design** (`scenesmith/floor_plan_agents/stateful_floor_plan_agent.py`): The Floor Plan Agent designs the architectural skeleton of the scene. It manages room placement, wall geometry, and openings like doors and windows, creating a valid architectural layout based on the text prompt.
- **Stage 2: Furniture Placement** (`scenesmith/furniture_agents/stateful_furniture_agent.py`): Once the layout is ready, the Furniture Agent takes over. It identifies functional zones and places primary objects (tables, chairs, cabinets). It balances aesthetic arrangement with physical constraints like reachability and clearance.
- **Stage 3: Wall & Ceiling Accents** (`scenesmith/wall_agents/stateful_wall_agent.py`): Decorative and functional secondary items are added next. The Wall Agent manages items like mirrors and artwork, while the Ceiling Agent places lighting fixtures, ensuring they are logically aligned with the furniture layout.
- **Stage 4: Manipuland Arrangement** (`scenesmith/manipuland_agents/stateful_manipuland_agent.py`): The final placement stage focuses on 'manipulands'—small objects like fruits, books, or electronics. This agent identifies support surfaces on furniture and arranges these items to make the scene feel 'lived-in' and ready for robot interaction.
- **Generative 3D Asset Creation** (`scenesmith/agent_utils/geometry_generation_server/geometry_generation.py`): A core innovation of SceneSmith is on-demand asset generation. If an object isn't in the library, this server uses image-to-3D models (like SAM3D or Hunyuan3D) to generate a custom 3D mesh for the agent's specific needs.
- **Simulation Readiness** (`scenesmith/agent_utils/sdf_generator.py`): To be useful for robotics, scenes must be more than just meshes. This module generates Drake-compatible SDF files, including physical properties like mass, inertia, and friction, ensuring the scene is ready for physics-based simulation.
- **Robot Task Evaluation** (`scenesmith/robot_eval/success_validation/validator_agent.py`): Finally, SceneSmith includes a loop for verifying task success. The Validator Agent uses visual and geometric checks to confirm if a robot can successfully perform tasks (like 'place fruit on table') within the generated environment.
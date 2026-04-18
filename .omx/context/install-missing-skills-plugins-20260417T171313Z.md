Task statement
- Read AGENTS.md carefully and install the skills/plugins it mentions that are not currently available.

Desired outcome
- A grounded gap analysis between AGENTS.md-declared capabilities and actually installed local capabilities, followed by installation of missing items that are installable in this environment.

Stated solution
- Audit AGENTS.md, compare against local skills/plugins/runtime, then install missing components.

Probable intent hypothesis
- The user wants the local Codex/OMX environment aligned with the repository's expected orchestration surface so later workflows do not fail due to missing skills or plugins.

Known facts/evidence
- Repo-local AGENTS.md is a child document and inherits from higher-scope AGENTS instructions.
- `/root/.codex/AGENTS.md` explicitly lists workflow skills and runtime expectations.
- `~/.codex/skills` contains 270 installed skills.
- `omx doctor` reports:
  - Codex CLI installed
  - 20 agent prompts installed
  - 270 skills installed
  - 4 MCP servers configured
  - warning: no compatible packaged prebuilt or cargo found for Explore Harness
  - warning: legacy `~/.agents/skills` exists
  - warning: project `.omx/state` not created yet
- MCP resource listing is empty even though MCP servers are configured.
- Native agent TOMLs exist under `/root/.codex/agents/`.

Constraints
- Must follow deep-interview mode until the requirements boundary is clear.
- Should minimize unnecessary installs if capability already exists.
- No destructive cleanup without need.

Unknowns/open questions
- Whether “插件” means MCP servers, OMX runtime dependencies, Codex extensions, or all of them.
- Whether optional runtime warnings should be treated as mandatory install targets.

Decision-boundary unknowns
- Whether to install Rust/tooling to satisfy `omx explore`.
- Whether to clean legacy `~/.agents/skills`.
- Whether to run `omx setup --force` proactively.

Likely codebase touchpoints
- `/root/.codex/AGENTS.md`
- `/root/.codex/skills/`
- `/root/.codex/agents/`
- `/root/.codex/config.toml`
- `omx doctor`

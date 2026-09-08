## Project context

This is a personal fork of [jingyaogong/minimind](https://github.com/jingyaogong/minimind) for studying and tinkering with LLM training. The usual goal is to understand the implementation and the typical training process of LLM, test ideas, and keep useful notes. Producing a competitive model or preparing production software is not the default objective.

Favor work that is easy to inspect and explain. Preserve a clear relationship with upstream where practical so that experiments remain comparable and upstream changes can still be reviewed.

## Guidelines, rules, and conventions

### Working approach

- Prefer small, focused changes over new abstractions unless the task calls for them.
- Explain non-obvious training or modeling choices in the relevant study note or task handoff.
- Put durable findings in `study/notes/`. Update an existing topical note instead of creating a competing account of the same subject.
- Treat datasets, virtual environments, model weights, checkpoints, caches, and raw logs as generated artifacts. Check ignore rules before creating or downloading them, and do not commit them unless the user explicitly asks.

### Documentation and sources of truth

Read documentation selectively according to the task:

- Start with this file for fork-specific working rules.
- For learning guidance, study sessions, or hands-on exercises, read [the learning path](study/notes/learning-path.md) first. Keep its current progress and next steps up to date as the user works through the material.
- For intentional changes to upstream code, read [the local implementation changes](study/notes/local-implementation-changes.md). Add an entry when a local code change affects behavior or maintenance, and reference the implementation commit.
- Read the relevant files under `study/notes/` for decisions and findings from this fork. For macOS setup or MPS work, read [the macOS and MPS setup note](study/notes/macos-mps-setup.md).
- Use `README.md` or `README_en.md` for upstream concepts, workflows, and background when needed.

The README files come from upstream. They may contain repository URLs, environment assumptions, commands, results, or status claims that are wrong for this fork. Do not use them as evidence of this checkout's identity or behavior. Verify repository identity and branch state with Git, verify behavior in the current code, and treat fork-specific notes and history as the record of local decisions.

### Documentation style

- Write documentation canonically by default. Describe current requirements, decisions, behavior, and implementation without preserving transitional wording, discussion history, rejected alternatives, or decision-making chronology.
- Distinguish confirmed decisions from unresolved questions. Move resolved questions into the appropriate canonical document.
- Link to the owning document instead of repeating its content.
- Keep links valid when files are added, renamed, or reorganized.
- Remove obsolete references when behavior, scope, ownership, or meaning changes. If a past decision contains a durable lesson, express that lesson as a current rule or requirement.
- Preserve discussion or decision history only when the document is explicitly a history, changelog, decision log, or similar record.
- Let the editor or Markdown renderer wrap prose. Use line breaks only where Markdown structure requires them, such as headings, list items, code blocks, tables, or intentional paragraph breaks.

### Python environment and commands

- Use the repository-local `.venv`, which is managed with uv. Do not install project packages into a system, Homebrew, or pyenv interpreter.
- Do not assume activation persists between tool calls. Invoke `.venv/bin/python` and other environment executables directly. For package operations, use commands such as `uv pip install --python .venv/bin/python ...` and `uv pip check --python .venv/bin/python`.
- The training scripts expect to run from `trainer/`. For example: `cd trainer && ../.venv/bin/python train_pretrain.py --device mps`.
- On this Mac, pass `--device mps` explicitly until the current device-handling limitations are fixed. Read [the macOS and MPS setup note](study/notes/macos-mps-setup.md) before changing device, precision, or distributed-training behavior.
- If `.venv` is missing or fails validation, report the problem instead of silently using another interpreter.

### Version control

- Do not stage or commit changes unless the user explicitly asks. A request to implement an accepted detailed plan incorporates that plan's local commit authorization and pause points.
- Do not push commits or tags, dispatch remote workflows, or otherwise trigger a remote deployment unless the user explicitly asks.
- Preserve unrelated user changes and avoid destructive version-control commands.

### Review processes

After finishing a change set:

1. Run relevant local checks when applicable.
2. Use subagent(s) to review the changes.

Reviewer requirements:

- Reviewers must not edit files.
- Use the appropriate review skill(s):
  - Code changes: use the `code-review-readonly` skill.
  - UI-visible changes: use the `ui-audit-readonly` skill.
  - Documentation-only changes: use the `document-review-readonly` skill.
- If the change set is a mix of documentation and code changes, and/or the code changes are a mix of UI and non-UI, use all relevant review lenses. You can decide whether to use one reviewer (such as when the changes are small or tightly coupled) or to use multiple reviewers (such as when the documentation, UI, and non-UI parts are large or separable), each focusing on one lens.

Provide reviewers with:

- relevant project context,
- the specific changes being reviewed and the necessary context of those changes.
- any other supplementary info that you deem useful to the reviewer.
- checks run and results.

When reviewers return findings:

- Assess whether each finding is a real issue.
- If all findings are legitimate, fix them without asking.
- Pause and confirm with the user when:
	- you disagree with any of the findings.
	- addressing any findings would change things materially or add substantive new machinery.
	- there is anything that warrants the user's attention.
- After fixing, resume the same reviewer subagent(s) for re-review and provide only the delta: what changed, which findings were addressed, and checks rerun.
- Continue this review, fix, and re-review loop until there are no actionable findings or the user says to stop.
- Summarize the issues fixed and include the summary in your task completion report.

For later change sets:

- Follow the same review process described above (for both the first-pass and the second-pass reviews).
- Reuse prior reviewer subagents when continuity helps, and their context is still manageable. In this case, you don't need to provide the entire project context again, only the current change context.
- Start fresh reviewer subagents when the change set is unrelated to previous work, the old context is stale due to recent changes, you notice deterioration in subagents' responses or too many context compactions, or just any reason that you think a fresh reviewer would be cleaner.

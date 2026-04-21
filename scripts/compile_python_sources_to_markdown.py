from __future__ import annotations

import argparse
from pathlib import Path

ROOT_MARKER = "app.py"
ENGINE_DIR = "engine"
DEFAULT_OUTPUT = Path("report") / "compiled_python_sources.md"


def collect_source_files(project_root: Path) -> list[Path]:
    app_path = project_root / ROOT_MARKER
    engine_dir = project_root / ENGINE_DIR

    source_files: list[Path] = []
    if app_path.exists():
        source_files.append(app_path)

    if engine_dir.exists():
        source_files.extend(sorted(engine_dir.rglob("*.py")))

    return source_files


def render_markdown(project_root: Path, source_files: list[Path]) -> str:
    lines: list[str] = ["# Python Source Compilation", ""]
    lines.append(f"Project root: `{project_root}`")
    lines.append(f"Files included: {len(source_files)}")
    lines.append("")

    for source_path in source_files:
        relative_path = source_path.relative_to(project_root).as_posix()
        code = source_path.read_text(encoding="utf-8")
        lines.append(f"## {relative_path}")
        lines.append("")
        lines.append("```python")
        lines.append(code.rstrip())
        lines.append("```")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_output_path(project_root: Path, output_arg: str | None) -> Path:
    if output_arg:
        output_path = Path(output_arg)
        return output_path if output_path.is_absolute() else project_root / output_path
    return project_root / DEFAULT_OUTPUT


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile app.py and engine/*.py into a single markdown file."
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output markdown file path. Defaults to report/compiled_python_sources.md.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    source_files = collect_source_files(project_root)
    if not source_files:
        raise SystemExit("No Python source files found in app.py or engine/.")

    output_path = build_output_path(project_root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(project_root, source_files), encoding="utf-8")

    print(f"Wrote {len(source_files)} files to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import json
from pathlib import Path


def extract_code_cells():
    # Notebook name (same folder as this script)
    notebook_name = "river_temperature_dynamics_tutorial.ipynb"
    output_name = "river_temperature_dynamics_tutorial.py"

    notebook_path = Path(notebook_name)

    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    code_lines = []
    code_lines.append("# Auto-generated from Jupyter Notebook\n")
    code_lines.append("# Source: river_temperature_dynamics_tutorial.ipynb\n\n")

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])

            code_lines.append("\n# ===== Cell Separator =====\n")

            if isinstance(source, list):
                lines = source
            else:
                lines = [source]

            for line in lines:
                # Skip Jupyter magic commands
                if line.strip().startswith("%"):
                    continue
                code_lines.append(line)

            code_lines.append("\n")

    with open(output_name, "w", encoding="utf-8") as f:
        f.writelines(code_lines)

    print(f"✅ Extracted code written to: {output_name}")


if __name__ == "__main__":
    extract_code_cells()
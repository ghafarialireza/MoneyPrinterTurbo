import ast
import os
import subprocess
import symtable
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"


class TestWebuiModuleReferences(unittest.TestCase):
    """The WebUI keeps its service modules reachable from every callback."""

    def test_no_callback_shadows_an_imported_service_module(self):
        """
        A local named after an imported module breaks the whole function.

        Python decides a name is local for an entire scope, so one
        ``for material in ...`` loop makes every ``material.<call>`` in the same
        function an ``UnboundLocalError`` — including calls that run long before
        the loop. This already happened once: a provider-capability check added
        to the generation controls raised on a source the loop never touched, and
        only one specific sidebar state exposed it. Assert the shape instead of
        waiting for the branch to be exercised.
        """
        source = WEBUI_MAIN.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported.add(alias.asname or alias.name)

        offenders: list[str] = []

        def visit(table: symtable.SymbolTable, scope: tuple[str, ...]) -> None:
            for child in table.get_children():
                here = scope + (child.get_name(),)
                if child.get_type() == "function":
                    for symbol in child.get_symbols():
                        name = symbol.get_name()
                        if (
                            name in imported
                            and symbol.is_local()
                            and not symbol.is_parameter()
                        ):
                            offenders.append(
                                f"{'::'.join(here)} (line {child.get_lineno()}) "
                                f"rebinds imported name {name!r}"
                            )
                visit(child, here)

        visit(symtable.symtable(source, str(WEBUI_MAIN), "exec"), ())

        self.assertEqual(offenders, [], "\n".join(offenders))


class TestWebuiStartup(unittest.TestCase):
    def test_external_directory_prefers_project_app_package(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            conflicting_root = temp_path / "site-packages"
            conflicting_app = conflicting_root / "app"
            conflicting_app.mkdir(parents=True)
            (conflicting_app / "__init__.py").write_text(
                'source = "conflicting dependency"\n',
                encoding="utf-8",
            )

            env = os.environ.copy()
            existing_pythonpath = env.get("PYTHONPATH")
            env["PYTHONPATH"] = os.pathsep.join(
                part
                for part in (str(conflicting_root), existing_pythonpath)
                if part
            )
            script = textwrap.dedent(
                f"""
                from pathlib import Path
                from streamlit.testing.v1 import AppTest

                app = AppTest.from_file({str(WEBUI_MAIN)!r}, default_timeout=30)
                app.run()
                if app.exception:
                    raise RuntimeError([str(item.value) for item in app.exception])

                import app.config

                project_root = Path({str(ROOT_DIR)!r}).resolve()
                imported_config = Path(app.config.__file__).resolve()
                if project_root not in imported_config.parents:
                    raise RuntimeError(
                        f"app.config resolved outside project: {{imported_config}}"
                    )
                """
            )

            result = subprocess.run(
                [sys.executable, "-X", "utf8", "-c", script],
                cwd=temp_path,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=60,
            )

            self.assertEqual(
                result.returncode,
                0,
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )


if __name__ == "__main__":
    unittest.main()

"""Setup script for skyborn-spharm - Meson Fortran/C build with incremental Ninja caching."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.install_lib import install_lib
from wheel.bdist_wheel import bdist_wheel

CONDA_BUILD_MODE = os.environ.get("CONDA_BUILD") == "1"


def _active_python_prefix() -> Path:
    return Path(sys.prefix).resolve()


def configure_compiler_environment():
    if CONDA_BUILD_MODE:
        return
    if platform.system() == "Darwin" and "CC" not in os.environ:
        fc_path = shutil.which(os.environ.get("FC", "gfortran"))
        if fc_path and ("/opt/homebrew" in fc_path or "/usr/local" in fc_path):
            fc_dir = Path(fc_path).resolve().parent
            gcc_bins = sorted(fc_dir.glob("gcc-[0-9]*"))
            if not gcc_bins:
                gcc_bins = sorted(Path("/opt/homebrew/bin").glob("gcc-[0-9]*"))
            if gcc_bins:
                cc_bin = str(gcc_bins[-1])
                os.environ["CC"] = cc_bin
                cxx_bin = cc_bin.replace("/gcc-", "/g++-")
                if os.path.exists(cxx_bin):
                    os.environ.setdefault("CXX", cxx_bin)
    os.environ.setdefault("FC", "gfortran")
    os.environ.setdefault("F77", os.environ["FC"])
    os.environ.setdefault("F90", os.environ["FC"])
    os.environ.setdefault("CC", "gcc")


configure_compiler_environment()


class MesonBuildExt(build_ext):
    """Custom build_ext invoking Meson + Ninja with incremental inplace caching."""

    @staticmethod
    def _resolve_tool_command(executable_name: str, module_name: str, *args: str):
        executable_path = shutil.which(executable_name)
        if executable_path:
            return [executable_path, *args]
        return [sys.executable, "-m", module_name, *args]

    @staticmethod
    def _meson_command(*args: str):
        return MesonBuildExt._resolve_tool_command(
            "meson", "mesonbuild.mesonmain", *args
        )

    @staticmethod
    def _ninja_command(*args: str):
        return MesonBuildExt._resolve_tool_command("ninja", "ninja", *args)

    @staticmethod
    def _write_meson_native_file(build_dir: Path) -> Path:
        native_file = (build_dir / "meson-python-native.ini").resolve()
        python_executable = Path(os.path.abspath(sys.executable)).as_posix()
        native_file.write_text(
            "[binaries]\n"
            f"python = '{python_executable}'\n"
            f"build_python = '{python_executable}'\n",
            encoding="utf-8",
        )
        return native_file

    @staticmethod
    def _write_build_python_shim(build_dir: Path) -> Path:
        python_executable = shutil.which("python") or os.path.abspath(sys.executable)
        if os.name == "nt":
            shim_path = (build_dir / "build_python.bat").resolve()
            with shim_path.open("w", encoding="utf-8", newline="\r\n") as handle:
                handle.write("@echo off\r\n")
                handle.write(f'"{python_executable}" %*\r\n')
        else:
            shim_path = (build_dir / "build_python").resolve()
            python_executable = Path(os.path.abspath(python_executable)).as_posix()
            with shim_path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write("#!/bin/sh\n")
                handle.write(f'exec "{python_executable}" "$@"\n')
            shim_path.chmod(0o755)
        return shim_path

    def run(self):
        self.build_meson_modules()
        super().run()
        self.prune_installed_import_libraries()

    def prune_installed_import_libraries(self):
        if self.inplace or not getattr(self, "build_lib", None):
            return
        build_lib_path = Path(self.build_lib)
        if not build_lib_path.exists():
            return
        for import_lib in build_lib_path.rglob("*.dll.a"):
            import_lib.unlink()

    def _discover_meson_modules(self):
        pkg_root = Path("src") / "skybornspharm"
        modules = []
        for meson_file in sorted(pkg_root.rglob("meson.build")):
            rel_dir = meson_file.parent.relative_to(pkg_root)
            name = ".".join(rel_dir.parts) if rel_dir.parts else "skybornspharm"
            modules.append({"name": name, "path": meson_file.parent})
        return modules

    def build_meson_modules(self):
        for module in self._discover_meson_modules():
            self.build_meson_module(module)

    def _prepare_meson_env(self, build_dir: Path) -> dict[str, str]:
        env = os.environ.copy()
        active_prefix = _active_python_prefix()
        conda_prefix = str(active_prefix) if active_prefix.exists() else ""
        build_dir_path = str(build_dir.resolve())
        system = platform.system()

        if conda_prefix:
            env["CONDA_PREFIX"] = conda_prefix
            current_path = env.get("PATH", "")
            if system == "Windows":
                path_entries = [
                    build_dir_path,
                    conda_prefix,
                    os.path.join(conda_prefix, "Scripts"),
                    os.path.join(conda_prefix, "bin"),
                    os.path.join(conda_prefix, "Library", "bin"),
                    os.path.join(conda_prefix, "Library", "mingw-w64", "bin"),
                    os.path.join(conda_prefix, "Library", "usr", "bin"),
                ]
                path_entries = [entry for entry in path_entries if os.path.isdir(entry)]
                env["PATH"] = ";".join(path_entries + [current_path])
                if CONDA_BUILD_MODE:
                    toolchain = {
                        "CC": "x86_64-w64-mingw32-gcc.exe",
                        "CXX": "x86_64-w64-mingw32-g++.exe",
                        "FC": "x86_64-w64-mingw32-gfortran.exe",
                        "F77": "x86_64-w64-mingw32-gfortran.exe",
                        "F90": "x86_64-w64-mingw32-gfortran.exe",
                        "AR": "x86_64-w64-mingw32-ar.exe",
                        "RANLIB": "x86_64-w64-mingw32-ranlib.exe",
                    }
                    for key, exe in toolchain.items():
                        found = shutil.which(exe, path=env["PATH"])
                        if found:
                            env[key] = found
            elif system in ("Linux", "Darwin"):
                conda_bin = os.path.join(conda_prefix, "bin")
                path_entries = [build_dir_path]
                if os.path.isdir(conda_bin):
                    path_entries.append(conda_bin)
                env["PATH"] = ":".join(path_entries + [current_path])
                if system == "Linux" and (
                    CONDA_BUILD_MODE or os.path.exists(os.path.join(conda_bin, "gfortran"))
                ):
                    conda_lib = os.path.join(conda_prefix, "lib")
                    cur_lib = env.get("LD_LIBRARY_PATH", "")
                    env["LD_LIBRARY_PATH"] = (
                        f"{conda_lib}:{cur_lib}" if cur_lib else conda_lib
                    )
                if CONDA_BUILD_MODE and system == "Darwin":
                    for key in (
                        "AR",
                        "RANLIB",
                        "GCC_AR",
                        "GCC_RANLIB",
                        "LD",
                        "LDSHARED",
                        "CC_LD",
                        "CXX_LD",
                        "FC_LD",
                        "F77_LD",
                        "F90_LD",
                    ):
                        env.pop(key, None)
                    ar_path = shutil.which("llvm-ar", path=conda_bin) or shutil.which(
                        "ar", path="/usr/bin"
                    )
                    ranlib_path = shutil.which(
                        "llvm-ranlib", path=conda_bin
                    ) or shutil.which("ranlib", path="/usr/bin")
                    if ar_path or ranlib_path:
                        shim_dir = tempfile.mkdtemp(prefix="skybornspharm-tools-")
                        for nm in ("ar", "gcc-ar", "gar"):
                            if ar_path:
                                p = os.path.join(shim_dir, nm)
                                with open(p, "w", encoding="utf-8") as h:
                                    h.write(
                                        '#!/bin/sh\nif [ "x$1" = "x--version" ]; then echo "GNU ar (shim)"; exit 0; fi\n'
                                        f'exec "{ar_path}" "$@"\n'
                                    )
                                os.chmod(p, 0o755)
                        for nm in ("ranlib", "gcc-ranlib"):
                            if ranlib_path:
                                p = os.path.join(shim_dir, nm)
                                with open(p, "w", encoding="utf-8") as h:
                                    h.write(
                                        '#!/bin/sh\nif [ "x$1" = "x--version" ]; then echo "GNU ranlib (shim)"; exit 0; fi\n'
                                        f'exec "{ranlib_path}" "$@"\n'
                                    )
                                os.chmod(p, 0o755)
                        env["PATH"] = f"{shim_dir}:{env['PATH']}"
        else:
            current_path = env.get("PATH", "")
            path_sep = ";" if system == "Windows" else ":"
            env["PATH"] = (
                f"{build_dir_path}{path_sep}{current_path}"
                if current_path
                else build_dir_path
            )
        return env

    def build_meson_module(self, module):
        module_path = module["path"]
        build_dir = module_path / "build"
        clean_requested = os.environ.get("SKYBORN_CLEAN_BUILD") == "1"
        can_reuse = (
            bool(self.inplace)
            and not clean_requested
            and (build_dir / "build.ninja").exists()
        )
        env = self._prepare_meson_env(build_dir)

        if not can_reuse:
            if build_dir.exists():
                shutil.rmtree(build_dir)
            build_dir.mkdir(parents=True, exist_ok=True)
            native_file = self._write_meson_native_file(build_dir)
            self._write_build_python_shim(build_dir)

            setup_cmd = self._meson_command(
                "setup",
                "build",
                ".",
                "--buildtype=release",
                "--native-file",
                str(native_file),
            )
            setup_cmd.append(
                "-Db_lto=false" if (CONDA_BUILD_MODE or not self.inplace) else "-Db_lto=true"
            )
            if not self.inplace and getattr(self, "build_lib", None):
                build_lib_path = Path(self.build_lib).resolve()
                setup_cmd.extend(
                    [
                        f"--python.purelibdir={build_lib_path}",
                        f"--python.platlibdir={build_lib_path}",
                    ]
                )
            subprocess.run(setup_cmd, cwd=str(module_path), check=True, env=env)

        subprocess.run(
            self._meson_command("compile", "-C", "build"),
            cwd=str(module_path),
            check=True,
            env=env,
        )

        if not self.inplace and getattr(self, "build_lib", None):
            subprocess.run(
                self._meson_command("install", "-C", "build", "--only-changed"),
                cwd=str(module_path),
                check=True,
                env=env,
            )
            self.prune_installed_import_libraries()


class CustomDevelop(develop):
    def run(self):
        self.run_command("build_ext")
        super().run()


class CustomInstall(install):
    def run(self):
        self.run_command("build_ext")
        super().run()


class CustomInstallLib(install_lib):
    """Prune transient build artifacts, sources, and bytecode from installed tree."""

    _PRUNE_DIR_NAMES = {
        "__pycache__",
        "build",
        "build_local",
        "meson-private",
        "meson-logs",
        "meson-info",
        "src",
    }
    _PRUNE_FILES = {"build_split_pyf.py"}

    def run(self):
        super().run()
        package_root = Path(self.install_dir) / "skybornspharm"
        if not package_root.exists():
            return
        for fname in self._PRUNE_FILES:
            target = package_root / fname
            if target.exists():
                target.unlink()
        for path in sorted(
            package_root.rglob("*"), key=lambda item: len(item.parts), reverse=True
        ):
            if path.is_dir() and (
                path.name in self._PRUNE_DIR_NAMES or path.name.startswith("build_")
            ):
                shutil.rmtree(path, ignore_errors=True)
            elif path.is_file() and path.suffix.lower() in {".pyc", ".pyo"}:
                path.unlink()


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class CustomBdistWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False


if __name__ == "__main__":
    setup(
        cmdclass={
            "bdist_wheel": CustomBdistWheel,
            "build_ext": MesonBuildExt,
            "develop": CustomDevelop,
            "install": CustomInstall,
            "install_lib": CustomInstallLib,
        },
        distclass=BinaryDistribution,
        ext_modules=[],
    )

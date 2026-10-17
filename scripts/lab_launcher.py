#!/usr/bin/python3
"""Desktop entry point for the local π0.5 simulation lab."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
TITLE = "π0.5 仿真实验室"


def notify(message, *, error=False):
    """Keep launch failures visible when a desktop entry has no terminal."""
    print(message, file=sys.stderr if error else sys.stdout)
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        try:
            (messagebox.showerror if error else messagebox.showinfo)(TITLE, message, parent=root)
        finally:
            root.destroy()
    except Exception:
        # The printed message remains available for terminal/headless invocations.
        pass


def desktop_quote(value):
    """Quote one Exec argument using Desktop Entry string and Exec escaping."""
    value = str(value)
    if "\n" in value or "\r" in value:
        raise ValueError("快捷方式路径不能包含换行")
    value = value.replace("%", "%%").replace("\\", "\\\\\\\\")
    for character in ('"', '$', '`'):
        value = value.replace(character, "\\\\" + character)
    return '"' + value + '"'


def install_shortcut(root=ROOT):
    desktop = Path(subprocess.check_output(["xdg-user-dir", "DESKTOP"], text=True).strip())
    if not desktop.is_absolute():
        raise ValueError("xdg-user-dir 未返回有效桌面目录")
    desktop.mkdir(parents=True, exist_ok=True)
    shortcut = desktop / "pi05-simulation-lab.desktop"
    command = " ".join(desktop_quote(value) for value in
                       ("/usr/bin/python3", root / "scripts/lab_launcher.py"))
    if shortcut.is_symlink() or (shortcut.exists() and
                                f"Exec={command}" not in shortcut.read_text(encoding="utf-8").splitlines()):
        raise FileExistsError(f"保留已有的其它快捷方式：{shortcut}")
    shortcut.write_text(
        "[Desktop Entry]\nType=Application\nVersion=1.0\n"
        f"Name={TITLE}\nComment=模型、训练与仿真实验工作台\nExec={command}\n"
        "Icon=applications-science\nTerminal=false\nCategories=Education;Science;\n",
        encoding="utf-8",
    )
    shortcut.chmod(0o755)
    try:
        subprocess.run(["gio", "set", str(shortcut), "metadata::trusted", "true"], check=True,
                       timeout=5, capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError) as error:
        print(f"快捷方式已安装，但桌面信任标记未设置：{error}", file=sys.stderr)
    return shortcut


def launch_lab(root=ROOT):
    log_directory = root / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    with (log_directory / "lab-launcher.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            notify("仿真实验室已在运行，请切换到已有窗口。")
            return 0
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=log_directory,
            prefix=f"lab-launcher-{time.strftime('%Y%m%d-%H%M%S')}-", suffix=".log", delete=False,
        ) as log:
            log_path = Path(log.name)
            command = [str(root / ".venv-sim/bin/python"), "-u", "scripts/simulation_lab.py"]
            log.write(f"Working directory: {root}\nCommand: {command!r}\n")
            log.flush()
            try:
                # The GUI also retains the lock if this waiting wrapper is interrupted.
                result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                        pass_fds=(lock.fileno(),))
                returncode = result.returncode
            except OSError as error:
                log.write(f"Launcher error: {error}\n")
                returncode = 1
        if returncode:
            notify(f"仿真实验室退出（代码 {returncode}）。\n请查看日志：\n{log_path}", error=True)
        return returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-shortcut", action="store_true", help="安装可信桌面快捷方式，不启动界面")
    args = parser.parse_args(argv)
    try:
        if args.install_shortcut:
            print(f"已安装桌面快捷方式：{install_shortcut()}")
            return 0
        return launch_lab()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        notify(f"无法启动或安装仿真实验室：{error}", error=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

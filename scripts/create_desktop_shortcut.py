"""
scripts/create_desktop_shortcut.py
AEGIS – Label-Free Threat Detection in One-Way Network Traffic (SIH26145)

Utility to create or update the Windows Desktop shortcut for AEGIS:
- Detects the current Windows user's Desktop (including redirected / OneDrive paths).
- Targets 'START_AEGIS.bat' in the project root.
- Sets the working directory to the project root.
- Uses standard Windows system icons (network/security).
- Idempotent: avoids creating duplicate shortcuts.
- Zero external package dependencies (uses native Windows Script Host via PowerShell).
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def get_desktop_dir() -> Path:
    """
    Detects the current user's Desktop directory.
    Queries Windows Shell API via PowerShell to correctly handle OneDrive
    redirection, custom profile locations, and localized folders.
    """
    # 1. Preferred: Query Windows Shell SpecialFolder for Desktop
    try:
        ps_query = "[Environment]::GetFolderPath('Desktop')"
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", ps_query],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        detected_path = proc.stdout.strip()
        if detected_path:
            p = Path(detected_path)
            if p.exists() and p.is_dir():
                return p
    except Exception:
        pass

    # 2. Fallback: %USERPROFILE%\Desktop
    user_profile = os.environ.get("USERPROFILE")
    if user_profile:
        p = Path(user_profile) / "Desktop"
        if p.exists() and p.is_dir():
            return p

    # 3. Fallback: Path.home() / Desktop
    home_desktop = Path.home() / "Desktop"
    if home_desktop.exists() and home_desktop.is_dir():
        return home_desktop

    # If none exist on disk, return the standard path so it can be created if needed
    return Path.home() / "Desktop"


def resolve_icon(project_root: Path) -> str:
    """
    Selects a suitable icon:
    1. Looks for any project-specific .ico file.
    2. Falls back to standard Windows shell32.dll network/monitor icon (index 18).
    """
    # Check for project icon if present
    for candidate in [project_root / "assets" / "aegis.ico", project_root / "aegis.ico"]:
        if candidate.exists():
            return str(candidate.resolve())

    # Standard Windows Network / Communication Tap icon from shell32.dll
    sys_root = os.environ.get("SystemRoot", r"C:\Windows")
    shell32 = Path(sys_root) / "System32" / "shell32.dll"
    if shell32.exists():
        return f"{shell32.resolve()},18"

    return ""


def create_or_update_shortcut(force: bool = False) -> bool:
    """
    Creates or updates the Windows Desktop shortcut named 'AEGIS – Threat Detection.lnk'.
    Returns True on success, False on failure.
    """
    project_root = Path(__file__).resolve().parents[1]
    target_bat = project_root / "START_AEGIS.bat"

    if not target_bat.exists():
        print(f"[ERROR] Target batch file not found at: {target_bat}")
        print("Please ensure 'START_AEGIS.bat' exists in the project root.")
        return False

    desktop_dir = get_desktop_dir()
    if not desktop_dir.exists():
        print(f"[ERROR] Desktop directory could not be located or does not exist: {desktop_dir}")
        return False

    shortcut_name = "AEGIS – Threat Detection.lnk"
    shortcut_path = desktop_dir / shortcut_name
    already_exists = shortcut_path.exists()

    if already_exists and not force:
        print(f"[INFO] Shortcut already exists at:")
        print(f"       {shortcut_path}")
        print("       Updating target and working directory to ensure synchronization...")

    icon_spec = resolve_icon(project_root)
    description = "AEGIS - Label-Free Threat Detection in One-Way Network Traffic (SIH26145)"

    # PowerShell script to create/update shortcut via WScript.Shell COM
    # Escapes single quotes in paths for PowerShell safety
    ps_shortcut_path = str(shortcut_path.resolve()).replace("'", "''")
    ps_target_bat = str(target_bat.resolve()).replace("'", "''")
    ps_working_dir = str(project_root.resolve()).replace("'", "''")
    ps_desc = description.replace("'", "''")
    ps_icon = icon_spec.replace("'", "''")

    ps_commands = [
        "$WshShell = New-Object -ComObject WScript.Shell",
        f"$Shortcut = $WshShell.CreateShortcut('{ps_shortcut_path}')",
        f"$Shortcut.TargetPath = '{ps_target_bat}'",
        f"$Shortcut.WorkingDirectory = '{ps_working_dir}'",
        f"$Shortcut.Description = '{ps_desc}'",
    ]
    if ps_icon:
        ps_commands.append(f"$Shortcut.IconLocation = '{ps_icon}'")
    ps_commands.append("$Shortcut.Save()")

    ps_script = "; ".join(ps_commands)

    try:
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to create shortcut via PowerShell: {e}")
        if e.stderr:
            print(f"        Details: {e.stderr.strip()}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error creating shortcut: {e}")
        return False

    if shortcut_path.exists():
        status_action = "Updated" if already_exists else "Created"
        print("=" * 70)
        print(f"  [+] SUCCESS: {status_action} Desktop Shortcut!")
        print("=" * 70)
        print(f"  Name:             {shortcut_name}")
        print(f"  Location:         {shortcut_path}")
        print(f"  Target:           {target_bat}")
        print(f"  Working Dir:      {project_root}")
        print(f"  Icon Source:      {icon_spec or 'System Default'}")
        print("=" * 70)
        print("You can now launch AEGIS directly by double-clicking the desktop icon.")
        return True
    else:
        print(f"[ERROR] Shortcut file was not created at: {shortcut_path}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Create Windows Desktop Shortcut for AEGIS")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing shortcut")
    args = parser.parse_args()

    success = create_or_update_shortcut(force=args.force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

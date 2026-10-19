"""Restore the fetched task diff in a new clean worktree; never alter the verifier's index."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = "d1127ec4572b38bc55a8faf525a9b33c9fdda467"
BRANCH = "experiment/correction-learning"
VERIFIER = ROOT / ".local/backup-public-20261017/remote-checkout"


def git(directory, *arguments, data=None):
    return subprocess.run(["rtk", "proxy", "git", "-C", str(directory), *arguments],
                          input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=True).stdout


def main():
    remote = git(ROOT, "remote", "get-url", "origin").decode().strip()
    assert remote == "https://github.com/sencangWei/pi05-handson.git"
    head = git(ROOT, "rev-parse", "HEAD").decode().strip()
    git(VERIFIER, "fetch", "origin", BRANCH)
    fetched = git(VERIFIER, "rev-parse", "FETCH_HEAD").decode().strip()
    assert fetched == head
    destination = ROOT / ".local/negative-recovery-20261019" / f"final-restored-{head[:12]}"
    assert not destination.exists()
    git(VERIFIER, "worktree", "add", "--detach", str(destination), BASE)
    patch = git(VERIFIER, "diff", "--binary", "--full-index", BASE, fetched)
    git(destination, "apply", "--index", "--whitespace=nowarn", "-", data=patch)
    expected = git(VERIFIER, "rev-parse", f"{fetched}^{{tree}}").decode().strip()
    actual = git(destination, "write-tree").decode().strip()
    assert actual == expected
    paths = git(VERIFIER, "diff", "--name-only", "-z", "--diff-filter=ACMR", BASE, fetched).decode().split("\0")
    hashes = {}
    for name in filter(None, paths):
        reference = git(VERIFIER, "show", f"{fetched}:{name}")
        restored = (destination / name).read_bytes()
        assert restored == reference
        assert (ROOT / name).read_bytes() == reference, name
        hashes[name] = hashlib.sha256(restored).hexdigest()
    components = []
    for directory in [ROOT, ROOT / "openpi", ROOT / "openpi/third_party/libero"]:
        components.append({"directory": str(directory),
                           "git_root": git(directory, "rev-parse", "--show-toplevel").decode().strip(),
                           "origin": git(directory, "remote", "get-url", "origin").decode().strip(),
                           "head": git(directory, "rev-parse", "HEAD").decode().strip(),
                           "tracked_dirty": git(directory, "status", "--short", "--untracked-files=no").decode()})
    report = {"status": "PASS", "remote": remote, "branch": BRANCH,
              "base": BASE, "fetched_head": fetched, "expected_tree": expected,
              "restored_tree": actual, "files_verified": len(hashes), "file_sha256": hashes,
              "restored_directory": str(destination), "components": components,
              "method": "Fetch user remote, apply full binary diff to a fresh detached base tree, compare full Git tree and every delivered file with both fetched commit and working file. No reset/clean of existing index.",
              "weight_backup": "Separate remote-adapter-verification.json proves inference recovery; not optimizer or raw data backup."}
    output = ROOT / ".local/negative-recovery-20261019/final-remote-restore-proof.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["status", "fetched_head", "restored_tree", "files_verified"]}))


if __name__ == "__main__":
    main()

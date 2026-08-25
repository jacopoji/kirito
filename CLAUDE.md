# Working with the GitHub CLI (`gh`) in this repo

This repo was originally created from `sasanquaa/komari`, which is now archived, then renamed
from `komari` to `kirito`, and was later transferred from `jgormigon/kirito` to `jacopoji/kirito`
(now private, on a GitHub Pro plan for the higher Actions minutes quota). A git remote named
`upstream` still points at the archived repo:

```
origin    https://github.com/jacopoji/kirito.git
upstream  https://github.com/sasanquaa/komari.git
```

`gh`'s default repo-resolution heuristic treats a remote literally named `upstream` as the
"parent"/base repo by convention and prefers it over `origin`, regardless of GitHub's actual fork
relationship (GitHub's API reports `jacopoji/kirito` as `isFork: false`, so this isn't a real fork
link - it's purely `gh`'s local remote-name heuristic). That means unqualified `gh` commands run in
a fresh clone - notably `gh pr create` - silently target `sasanquaa/komari` instead of
`jacopoji/kirito`, and fail with:

```
pull request create failed: GraphQL: Repository was archived so is read-only (createPullRequest)
```

**Fix (one-time per clone/worktree):**

```bash
gh repo set-default jacopoji/kirito
```

This writes `remote.origin.gh-resolved = base` into the repo's git config, which is shared across
all worktrees of a given clone. After running it once, plain `gh pr create`, `gh pr view`, `gh repo
view`, etc. all correctly resolve to `jacopoji/kirito`.

If you hit the archived-repo error before running the above (or don't want to change local config),
pass `--repo jacopoji/kirito` explicitly to the `gh` command, or fall back to the REST API directly:

```bash
gh api repos/jacopoji/kirito/pulls -f title="..." -f head="branch-name" -f base="master" -f body="..."
```

Old clones/worktrees still pointing `origin` at `jgormigon/kirito.git` will keep working via
GitHub's automatic redirect, but should have their remote updated with
`git remote set-url origin https://github.com/jacopoji/kirito.git` to avoid relying on it, since
the previous owner (`jgormigon`) only retains collaborator (push) access, not ownership.

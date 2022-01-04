Release Instructions
====================

Versions should take the form "v<major>.<minor>.patch". For example, "v0.3.0" is a valid
version, while "v1" is not and "0.3.0" is not.

1. Make sure all PRs are merged and tests pass.

2. Prepare a release branch with `git checkout -b release/util/<version>`.

3. Update the HISTORY.md, replacing the "latest" version heading with the new version.

4. Commit your changes so far to the release branch.

5. In the pace-util directory, run `bumpversion <major/minor/patch>`. This will create a new commit.

6. `git push -u origin release/util/<version>` and create a new pull request in Github.

7. When the pull request is merged to main, `git checkout main` and `git pull`,
   followed by `git tag util/<version>`.

8. Run `git push origin --tags` to push all local tags to Github.

9. Run `make release` to push latest release to PyPI. Contact a core developer to get the
   necessary API token.

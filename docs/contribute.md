# Contribution guidelines

Following these guidelines helps to communicate that you respect the time of the developers managing and developing this project.

**Table of contents**

- [Branching model](#branching-model)
- [Feature branches](#feature-branches)
- [Write a feature](#write-a-feature)
- [Merge to staging](#merge-to-staging)
- [Merge to production](#merge-to-production)
- [Best practices](#best-practices)

## Branching model

Instead of a single main branch, we use two branches to record the history of the project:

- `main`: production branch.
- `develop`: development and default branch.

![how-it-works](01%20How%20it%20works.svg)

The production branch `main` stores the official release history, and the `develop` branch serves as an integration branch for new features and bug fixes.

## Feature branches

Each new feature must reside in its own branch and should be called `feature/{your-name}`.

![Feature-branches](02%20Feature%20branches.svg)

But, instead of branching off of `main`, feature branches use `develop` as their parent branch.

When a feature is complete, [it gets merged back into `develop`](#merge-to-staging). Features must never interact directly with `main`.

## Write a feature

1. **Create a new feature branch** based off `develop`.

   ```console
   git checkout develop
   git pull
   git checkout -b {branch-name}
   git push --set-upstream origin {branch-name}
   ```

2. **If other contributors, rebase frequently** to incorporate upstream changes from `develop` branch.

   ```console
   git fetch origin
   git rebase origin/develop
   ```

3. When feature is complete and tests pass, **stage and commit the changes**.

   ```console
   git add --all
   git status
   git commit --verbose
   ```

4. **Write a good commit message**.

5. **Publish changes to your branch**.

   ```console
   git push
   ```

## Merge to staging

1. **If you've created more than one commit**, squash them into into cohesive commits with good messages.
   Then, you would [rebase interactively](https://help.github.com/articles/about-git-rebase/):

   ```console
   git fetch origin
   git rebase -i HEAD~5
   # Rebase the commit messages
   git push --force-with-lease origin {branch-name}
   ```

2. **Merge changes** from your feature branch to `develop`.

   ```console
   git checkout develop
   git merge <branch-name> --ff-only
   git push
   ```

3. **Delete the feature branch** local and remote.

   ```console
   git push origin --delete <branch-name>
   git branch -D <branch-name>
   ```

## Merge to production

Every time we deploy a new version to the production environment the `main` branch needs to be merged from `develop` branch.

```console
git checkout main
git pull
git merge develop --ff-only
git push
```

## Best practices

- [All `feature` branches](#feature-branches) are created from `develop`.
- All `feature` branches are named `feature/{title}`.
- Rebase frequently your `feature` branches to incorporate upstream changes from `develop`.
- [Squash multiple trivial commits](https://help.github.com/articles/about-git-rebase/) into a single commit, before merging to `develop`.
- Write a good commit message.
- When a `feature` is complete [it is merged](#merge-to-staging) into the `develop`.
- Avoid merge commits by using a rebase workflow.
- Avoid working on `main` branch except as described below.
- Keep the [CHANGELOG.md](../CHANGELOG.md) file updated anytime you plan to merge changes to any of the main branches.


# Contributing

## Development workflow

Hi there! This repository follows the [GitHub flow](https://docs.github.com/en/get-started/quickstart/github-flow). The GitHub flow contains the main branch and many feature branches. Generally speaking, the main branch always uses no direct commit and only can be integrated by rebase and merge. The feature branches, like new features, bug fixes, refactoring, experiments, etc., are used for development. The GitHub flow keeps the main branch working well with documents and tests.

## Commit

This repository uses the [Angular commit style](https://github.com/angular/angular.js/blob/master/DEVELOPERS.md#commit-message-format), which looks like this:

```shell
<type>(optional scope): short summary in present tense

(optional body: explains motivation for the change)

(optional footer: note BREAKING CHANGES here, and issues to be closed)
```

Generally speaking, you need to at least specify a type and a short summary for each commit. `<type>` refers to the kind of change made and is usually one of:

- `feat`: A new feature.
- `fix`: A bug fix.
- `docs`: Documentation changes.
- `style`: Changes that do not affect the meaning of the code (white space, formatting, missing semi-colons, etc).
- `refactor`: A code change that neither fixes a bug nor adds a feature.
- `perf`: A code change that improves performance.
- `test`: Changes to the test framework.
- `build`: Changes to the build process or tools.

By using the standardized commit message in this Angular commit style, the continuous integration configuration will automatically bump version numbers based on keywords it finds in commit messages.

## References

- [Git for Professionals Tutorial - Tools & Concepts for Mastering Version Control with Git](https://www.youtube.com/watch?v=Uszj_k0DGsg)
- [GitHub flow](https://docs.github.com/en/get-started/quickstart/github-flow)
- [How to Write a Git Commit Message](https://cbea.ms/git-commit/)
# Contributing to SALT

We welcome contributions to SALT. Please read the following guidelines before submitting your pull request.

### Examples of contributions include:

- Bug fixes
- Documentation improvements
- Tests

## Reporting issues

Go to this repository's [issues page](https://github.com/AI21Labs/salt/issues) and click on the "New Issue" button.


Include the following information in your post:

- Describe what you expected to happen.
- If possible, include a [minimal reproducible example](https://stackoverflow.com/help/minimal-reproducible-example) to help us
  identify the issue. This also helps check that the issue is not with
  your own code.
- Describe what actually happened. Include the full traceback if there
  was an exception.
- List your Python version. If possible, check if this
  issue is already fixed in the latest releases or the latest code in
  the repository.

## Submit a pull request

Fork the SALT repository and clone it to your local machine. Create a new branch for your changes:

    git clone https://github.com:AI21Labs/USERNAME/salt
    cd salt
    git checkout -b my-fix-branch master

### Installation

We recommend creating your own virtual environment using pyenv or virtualenv, in order to eliminate unnecessary dependencies from external libraries.

You can use [pyenv](https://github.com/pyenv/pyenv) to set up your environment:

    pyenv install 3.9.0
    pyenv virtualenv 3.9.0 salt
    pyenv activate salt
    pip install -r requirements.txt

After that Install [pre-commit](https://pre-commit.com/#installation) and run:

    pre-commit install --install-hooks -t pre-commit -t commit-msg

Installing the pre-commit hooks would take care of formatting and linting your code before committing.
Please make sure you have the pre-commit hooks installed before committing your code.


### Commits

Each commit should be a single logical change and should be aligned with the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.
Since we are using a pre-commit hook to enforce this, any other commit message format will be rejected.

### How to open a pull request?

Push your branch to your forked repository and open a pull request against the `main` branch of the SALT repository. Please make sure to include a description of your changes in the pull request.

The title of the pull request should follow the above-mentioned [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.

### Feedback

If you have any questions or feedback, please feel free to reach out to us.

We appreciate and encourage any contributions to SALT. Please take the reviewer feedback positively and make the necessary changes to your pull request.

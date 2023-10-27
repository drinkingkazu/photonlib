# Contribution guide

In this guide you will get an overview of the contribution workflow from opening an issue, creating a PR, reviewing, and merging the PR.

## Making changes

1. Fork the repository

Fork the repo so you can make your changes without affecting the original project until you're ready to merge. Go to the repository website, then click "fork". You can fork a repository into your personal github space. For example, the original repository is at https://github.com/CIDeR-ML/photonlib, and Kazu's work is at https://github/drinkingkazu/photonlib which is in my github space.

Then you can simply clone:
```
$ git clone git@github.com:drinkingkazu/photonlib
```

**Use YOUR fork, not Kazu's!** :D

2. Create a working branch from `develop`

Two key branches to remember are `main` and `develop`. The `main` branch is used for releasing software tags. The `develop` branch is used for merging development work by collaborators. You should not work directly on neither of these branches.

If you're working on a substantial change to the repository that's experimental in nature, create a new branch for your changes from the `develop`. Name your branch something descriptive instead of something like `kazu-io-bugfix` which I used in the example below. 

```
$ cd photonlib
$ git checkout develop
$ git checkout -b kazu-io-bugfix
```

## Committing changes

Before committing, make sure to test your changes to ensure you aren't breaking anything. See [Test running and writing](#Test-Running-and-Writing).

If you're adding a new feature, make sure to add tests for it. If you're fixing a serious bug, make sure to add a test that would have caught the bug.

Commit your changes once you're happy with them. Because multiple people work on the project, be sure to include a **descriptive** commit message that explains your changes, and to not include any files that aren't related to your change.

## Merging into local `develop`

Now htat you committed your changes to your branch, the next step is to merge them into the **local** `develop` branch. 

```
$ git checkout develop
$ git merge kazu-io-bugfix
```

**Then test again**. If the test passes, then your branch is compatible and good to go with your local `develop` branch. If anything goes wrong, you can now directly make fixes in the `develop` branch. Note any issue is unlikely here unless you updated your `develop` branch while working on the development.

## Merging the remote `develop`

When you're finished with your changes, make sure you pull the remote `develop` branch from the original repository to get the latest changes, then merge them to your branch in your fork. This is different from the previous step (i.e. merging the "local" `develop` branch).

```
$ git remote add cider https://github.com/CIDeR-ML/photonlib
$ git fetch cider
$ git merge cider/develop
```

 **Then test again** and make sure to fix any new issues found. 

## Opening a Pull Request (PR)
Open a pull request (on the github webpage, from your fork) to the original repository. Make sure to include a title and description that explains your changes.

You should review your own PR first before others do. Double check that all commits have descriptive titles, that you've included tests for any new features or serious bug fixes, and that there are no random files changed.

After you review your own PR, wait for at least one other person to take a look at your changes and approve them.

## Test running and writing

Tests are written using the [pytest](https://docs.pytest.org/en/latest/) framework. To run the tests, run the following command:

```bash
cd photonlib
pytest tests/
```

To write tests, create a new file in the `tests/` directory. The directory follows the same structure as the package directory, so make sure to create the file in the correct subdirectory. For example, if you're testing a function in `pfmatch/algorithm`, create your file in `tests/algorithm`. Name the file something descriptive with a `test_` prefix. Then, write your tests using pytest. A sample test file is shown below:

```python
from (...) import l2_norm

import pytest
from tests.fixtures import rng

# pytest function
def test_l2_norm(rng):
    # test that weighted unit vectors should have norms
    # equal to the weights
    sqrt3 = np.sqrt(3)
    norm_1 = [1/sqrt3, 1/sqrt3, 1/sqrt3]

    pos_length = int(rng.random()*98) + 2
    pos = np.tile(norm_1, (pos_length, 1)) # Nx3 array

    weights = rng.random(size=(pos_length, 1))
    pos *= weights
    assert np.allclose(l2_norm(pos), weights.ravel()), \
            "expected sum of weights to equal sum of weighted unit vector norms"

    # test exception raises
    with pytest.raises(ValueError):
        l2_norm(1) # not a vector

    ...
```

Note that all test functions need to start with `test_`. See the [pytest documentation](https://docs.pytest.org/en/latest/) for more information.

### Note:
* if you are using random numbers like in the above example, import the `rng` generator from `tests.fixtures` to ensure that the tests are reproducible. Add `rng` as an argument to your test function and use it as you would a normal `np.random` generator. Pytest will automatically give it to your function. We use this to ensure that all code that uses a pseudo-random number generator has the same seed, so as to make all tests reproducable. (A [fixture](https://docs.pytest.org/en/latest/fixture.html) is a fancy name for a function that runs before all the tests and returns a value that can be used as an argument in your tests.)
* For random numbers in `pytorch`, import `torch_rng` from `tests.fixtures`, and use it as the `generator` argument to whatever random function you're using, i.e., `torch.rand(..., generator=torch_rng)`.
* Avoid running direct comparisons between float numbers. Instead, use `np.allclose` to check if two arrays are close to each other. This is because of the way that floating point numbers are stored in memory. See [this](https://docs.python.org/3/tutorial/floatingpoint.html) for more information.
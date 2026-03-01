Documentation Maintenance
=========================

Update API Function Documentation
---------------------------------

To refresh all module/function pages from the current code, run from ``pyologger/``:

.. code-block:: bash

   sphinx-apidoc -f -e -o docs/source pyologger

This regenerates ``docs/source/pyologger*.rst`` files used by autodoc.

Build The Docs Locally
----------------------

From ``pyologger/docs``:

.. code-block:: bash

   make clean
   make html

Open:

``docs/build/html/index.html``

Deploying Docs
--------------

You do **not** need to publish to PyPI to deploy docs.

GitHub Pages (current project setup):

1. Push to ``main`` with docs changes.
2. GitHub Actions workflow ``Deploy Docs (GitHub Pages)`` builds ``docs/build/html``.
3. The workflow deploys using Pages artifact + ``actions/deploy-pages``.
4. In repository settings, set Pages source to ``GitHub Actions``.

Suggested release flow:

1. Regenerate API rst files with ``sphinx-apidoc``.
2. Build locally with ``make html`` and fix warnings/errors.
3. Push branch and validate docs in GitHub Actions build logs.
4. Merge, then tag/release package separately if needed.

PyPI and docs deployment are independent steps.

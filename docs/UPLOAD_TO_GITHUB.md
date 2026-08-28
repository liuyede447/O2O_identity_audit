# Upload instructions

## Recommended repository

- Owner: `liuyede447`
- Repository name: `o2o-identity-audit`
- Visibility: Public after the authors approve the licence and file contents
- Description: `Reproducibility package for loss-active O2O identity stability in aerial tiny-object detection.`

## Browser upload

1. Create an empty GitHub repository with the name above.
2. Do not create an additional README, `.gitignore`, or licence during setup;
   these files are already handled in this package.
3. Unzip the package locally.
4. Upload the **contents** of the `o2o-identity-audit-github-v1.0.0` directory,
   not the outer directory itself.
5. Commit with the message `Initial reproducibility release`.
6. Confirm that README links work and the **Cite this repository** panel appears.
7. Create a GitHub release tagged `v1.0.0`.

## Recommended DOI step

1. Link the GitHub account to Zenodo.
2. Enable the new repository in Zenodo's GitHub integration.
3. Create the GitHub `v1.0.0` release.
4. Wait for Zenodo to archive it and assign a DOI.
5. Replace the GitHub-only manuscript wording with the DOI wording in
   `MANUSCRIPT_LINK_TEXT.md`.

## Required author checks before public upload

- approve a software/data licence;
- confirm the three-author order and repository ownership;
- confirm that derived image identifiers and box coordinates may be shared;
- confirm that no model binary should be included;
- run `python scripts/verify_release.py` after any change.


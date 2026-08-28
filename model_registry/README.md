# Model artifact registry

`MODEL_ARTIFACTS_SHA256.csv` records the byte size and SHA-256 digest of the
pretrained initializations and selected author-trained checkpoints used during
the study.

The corresponding model binaries are intentionally not included. To verify a
locally obtained file:

```bash
python -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" /path/to/checkpoint.pt
```

Compare the printed digest with the appropriate row in the registry. A matching
filename without a matching digest is not the frozen artifact used in the
reported analysis.


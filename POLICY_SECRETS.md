# Policy Secrets

Policy secrets let you attach credentials (e.g. API keys) to a policy upload so they are
available at runtime during tournament episodes. Secrets are scoped per-policy and encrypted at
rest. Per-job S3 bundles are cleaned up after each job; the Secrets Manager entry persists for
the lifetime of the policy version.

## Usage

Pass `--secret-env KEY=VALUE` when uploading. You can repeat it for multiple keys:

```bash
cogames upload \
  -p ./my_policy -n my-llm-policy \
  --secret-env ANTHROPIC_API_KEY=sk-ant-... \
  --secret-env OTHER_SECRET=value
```

### Constraints

- Up to **10 keys** per policy
- Keys must match `[A-Z_][A-Z0-9_]*` (uppercase + underscores), max 100 chars
- Values max 2000 chars each, 64 KB total JSON payload
- Secrets are immutable per policy version -- upload a new version to change them

## Lifecycle

A secret passes through four stages between submission and cleanup:

### 1. Submit (CLI -> Backend)

The CLI sends `policy_secret_env` as part of the upload-complete request. The backend writes
the dict to **AWS Secrets Manager** keyed as `cogames/policy-secrets/<policy_version_id>`.
The secret is now at rest, tied to this specific policy version.

### 2. Dispatch (Backend -> S3)

When the backend creates a job (match), the dispatcher:

1. Reads each policy's secrets from Secrets Manager
2. Assembles a single job-scoped bundle: `{"policies": {"0": {"KEY": "val"}, "1": {...}}}`
3. Writes the bundle to **S3** (`s3://cogames-secrets/job-secrets/<job_id>/bundle.json`)
   encrypted with **SSE-KMS**
4. Generates a **presigned GET URL** (1-hour TTL) for that S3 object
5. Passes the URL to the runner pod as the `POLICY_SECRETS_URI` environment variable

The runner pod needs no AWS credentials to fetch the bundle -- the presigned URL is
self-contained.

### 3. Run (Runner -> Policy Subprocess)

The episode runner:

1. Fetches `POLICY_SECRETS_URI` via plain HTTPS (`urlopen`)
2. Parses the bundle JSON
3. Injects each policy's secrets into its subprocess via `Popen(env={**os.environ, **secrets})`

Each policy subprocess receives **only its own secrets**. Secrets never appear in logs or
artifacts.

### 4. Cleanup (Backend)

After a job completes or fails, the backend deletes the S3 bundle object. A 7-day S3
lifecycle rule acts as a safety net for any missed cleanups.

## Security Properties

- **Per-policy isolation**: each policy subprocess gets only its own env vars
- **Encrypted at rest**: SSE-KMS in S3, AWS-managed encryption in Secrets Manager
- **Short-lived access**: presigned URLs expire after 1 hour
- **No AWS credentials on runner**: the runner fetches via a plain HTTPS URL
- **Automatic cleanup**: S3 objects deleted after job completion + lifecycle fallback

### What this does NOT defend against

- Malicious code in a co-located policy reading `/proc` or shared memory (phase 2: container
  isolation)
- Secret rotation or updates (secrets are immutable per policy version)
- Audit logging of secret access (not yet implemented)

## Architecture Diagram

```
CLI                    Backend                    S3 (cogames-secrets)       Runner Pod
 |                       |                              |                       |
 |-- upload --secret-env |                              |                       |
 |   KEY=val             |                              |                       |
 |                       |-- CreateSecret               |                       |
 |                       |   (Secrets Manager)          |                       |
 |                       |                              |                       |
 |                       |  [job dispatched]            |                       |
 |                       |-- GetSecretValue             |                       |
 |                       |-- PutObject (SSE-KMS) ------>|                       |
 |                       |-- generate_presigned_url     |                       |
 |                       |-- set POLICY_SECRETS_URI ----|---------------------> |
 |                       |                              |                       |
 |                       |                              |<-- GET presigned URL --|
 |                       |                              |-- bundle.json ------->|
 |                       |                              |                       |
 |                       |                              |   Popen(env=secrets)  |
 |                       |                              |   [policy runs]       |
 |                       |                              |                       |
 |                       |  [job complete]              |                       |
 |                       |-- DeleteObject ------------->|                       |
```

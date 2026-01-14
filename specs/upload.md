# Upload Spec

Spec for `cogames upload` command.

## Goal

Users can successfully upload their own Python policies to the tournament server.

## Scope

**In scope:**

- Packaging and uploading user-provided Python policies
- Validating policy structure before upload
- Returning a policy ID on success
- Clear error messages on failure

**Out of scope:**

- Tournament handling after upload (scheduling, matchmaking, scoring)
- Policy execution correctness during matches
- Policy performance or resource limits during matches

## What is a Python Policy?

A Python policy is a class that implements `MultiAgentPolicy` from mettagrid.

### Requirements

1. **Implements MultiAgentPolicy**: Must subclass or implement the `MultiAgentPolicy` interface
2. **Dependencies**: Either relies on dependencies available in the tournament runtime (torch, numpy, etc.), or vendors
   them and includes in the upload bundle

## Checkpoint Bundles

A checkpoint bundle is a zip archive containing a `policy_spec.json` and associated files. See `SubmissionPolicySpec` in
`mettagrid/policy/submission.py` for the spec format.

## CLI Interface

```bash
# Upload with class path
cogames upload class=my_policy.MyPolicy -n my-name

# Upload with data path for weights
cogames upload class=my_policy.MyPolicy,data=./model.pt -n my-name

# Upload a checkpoint bundle
cogames upload file://./train_dir/checkpoint/ -n my-name

# Upload with a custom name
cogames upload class=my_policy.MyPolicy --name "my-agent-v2" -n my-name

# Include additional files
cogames upload class=my_policy.MyPolicy --include-files ./utils.py -n my-name
```

## Success Criteria

1. User can upload a valid Python policy implementing MultiAgentPolicy
2. User receives a policy ID they can use with `cogames submit`
3. Invalid policies are rejected with actionable error messages
4. Validation runs the policy for 10 steps in an isolated environment before upload

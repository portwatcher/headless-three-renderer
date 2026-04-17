#!/bin/zsh
REPO="portwatcher/headless-three-renderer"
BRANCH="main"
MAX_WAIT_SECONDS=1500 # 25 minutes
INTERVAL=45
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT_SECONDS ]; do
  RUN=$(gh run list --repo "$REPO" --branch "$BRANCH" --limit 1 --json databaseId,status,conclusion,displayTitle | jq -r '.[0]')
  ID=$(echo "$RUN" | jq -r '.databaseId')
  STATUS=$(echo "$RUN" | jq -r '.status')
  CONCLUSION=$(echo "$RUN" | jq -r '.conclusion')
  
  echo "Run $ID: status=$STATUS, conclusion=$CONCLUSION"
  
  if [ "$STATUS" = "completed" ]; then
    echo "Workflow completed with conclusion: $CONCLUSION"
    
    # List jobs
    JOBS=$(gh api "repos/$REPO/actions/runs/$ID/jobs" --jq '.jobs[] | {id, name, conclusion}')
    echo "$JOBS"
    
    # Check for failures
    FAILED_JOBS=$(echo "$JOBS" | jq -r 'select(.conclusion == "failure") | .id')
    for JOB_ID in ${(f)FAILED_JOBS}; do
      echo "Fetching logs for failed job $JOB_ID..."
      gh api "repos/$REPO/actions/jobs/$JOB_ID/logs" | tail -n 40
    done
    exit 0
  fi
  
  sleep $INTERVAL
  ELAPSED=$((ELAPSED + INTERVAL))
done

echo "Timed out waiting for workflow to complete."
exit 1

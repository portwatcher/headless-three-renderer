#!/bin/zsh
RUN_ID="24577816840"
REPO="portwatcher/headless-three-renderer"
MAX_WAIT_SECONDS=1500
INTERVAL=60
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT_SECONDS ]; do
  INFO=$(gh api "repos/$REPO/actions/runs/$RUN_ID" 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['status'],d.get('conclusion'))")
  STATUS=$(echo "$INFO" | awk '{print $1}')
  CONCLUSION=$(echo "$INFO" | awk '{print $2}')
  
  echo "Elapsed: ${ELAPSED}s - Status: $STATUS, Conclusion: $CONCLUSION"
  
  if [ "$STATUS" = "completed" ]; then
    echo "Workflow $RUN_ID completed."
    break
  fi
  
  sleep $INTERVAL
  ELAPSED=$((ELAPSED + INTERVAL))
done

if [ "$STATUS" != "completed" ]; then
  echo "Timed out."
  exit 1
fi

echo "Job Results:"
JOBS_RAW=$(gh api "repos/$REPO/actions/runs/$RUN_ID/jobs" 2>/dev/null)
echo "$JOBS_RAW" | python3 -c "import json,sys;[print(j['name'],j['conclusion'],j['id']) for j in json.load(sys.stdin)['jobs']]"

PUBLISH_JOB=$(echo "$JOBS_RAW" | jq -r '.jobs[] | select(.name == "Publish to npm")')
PUBLISH_CONCLUSION=$(echo "$PUBLISH_JOB" | jq -r '.conclusion')
PUBLISH_ID=$(echo "$PUBLISH_JOB" | jq -r '.id')

if [[ "$PUBLISH_CONCLUSION" == "failure" || "$PUBLISH_CONCLUSION" == "skipped" || "$PUBLISH_CONCLUSION" == "null" ]]; then
  echo "Publish job did not succeed (Conclusion: $PUBLISH_CONCLUSION). Fetching logs for ID $PUBLISH_ID..."
  gh api "repos/$REPO/actions/jobs/$PUBLISH_ID/logs" | tail -n 80
fi

echo "NPM view version:"
npm view headless-three-renderer version 2>&1

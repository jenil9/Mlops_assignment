#!/usr/bin/env bash
# Smoke test for the movie-recsys API.
# Usage: bash scripts/smoke_test.sh [BASE_URL]
# Default BASE_URL: http://localhost:8080
set -euo pipefail

BASE_URL="${1:-http://localhost:8080}"
PASS=0
FAIL=0

pass() { echo "PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "FAIL: $1"; exit 1; }

# 1. Wait for /health to return 200 (up to 120 s)
echo "=== 1. Waiting for /health ==="
WAITED=0
while true; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health" || echo "000")
    if [ "$STATUS" = "200" ]; then break; fi
    WAITED=$((WAITED + 2))
    if [ $WAITED -ge 120 ]; then fail "/health did not return 200 within 120s (last status: $STATUS)"; fi
    sleep 2
done
pass "/health returned 200"

# 2. /metrics and /reload return 404 through gateway
echo "=== 2. /metrics and /reload should return 404 ==="
METRICS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/metrics")
[ "$METRICS_STATUS" = "404" ] || fail "/metrics returned $METRICS_STATUS (expected 404)"
pass "/metrics returns 404"

RELOAD_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/reload")
[ "$RELOAD_STATUS" = "404" ] || fail "/reload returned $RELOAD_STATUS (expected 404)"
pass "/reload returns 404 through gateway"

# 3. /popular returns at least 10 movies
echo "=== 3. /popular returns >=10 movies ==="
POPULAR=$(curl -sf "$BASE_URL/popular" | jq '. | length')
[ "$POPULAR" -ge 10 ] || fail "/popular returned only $POPULAR movies"
pass "/popular returned $POPULAR movies"

# 4. Sign up and verify user_id >= 100000
echo "=== 4. Signup and /me ==="
RAND_SUFFIX=$(od -A n -t d -N 3 /dev/urandom | tr -d ' ')
USERNAME="smoke_${RAND_SUFFIX}"
PASSWORD="smoke-pass-1"

TOKEN=$(curl -sf -X POST "$BASE_URL/signup" \
    -H "Content-Type: application/json" \
    -d "{\"username\":\"$USERNAME\",\"password\":\"$PASSWORD\"}" \
    | jq -r '.token')
[ -n "$TOKEN" ] && [ "$TOKEN" != "null" ] || fail "signup did not return a token"
pass "signup returned token"

USER_ID=$(curl -sf "$BASE_URL/me" -H "Authorization: Bearer $TOKEN" | jq -r '.user_id')
[ "$USER_ID" -ge 100000 ] 2>/dev/null || fail "/me returned user_id=$USER_ID (expected >= 100000)"
pass "/me returned user_id=$USER_ID"

# 5. /recommend returns kind=popular for new user
echo "=== 5. /recommend returns kind=popular for new user ==="
KIND=$(curl -sf "$BASE_URL/recommend" -H "Authorization: Bearer $TOKEN" | jq -r '.kind')
[ "$KIND" = "popular" ] || fail "/recommend returned kind=$KIND (expected popular)"
pass "/recommend returned kind=popular"

# 6. Rate three movies
echo "=== 6. Rate three movies ==="
MOVIE_IDS=$(curl -sf "$BASE_URL/popular" | jq '[.[0:3] | .[] | .movie_id]')
M1=$(echo "$MOVIE_IDS" | jq '.[0]')
M2=$(echo "$MOVIE_IDS" | jq '.[1]')
M3=$(echo "$MOVIE_IDS" | jq '.[2]')

for MID_RATING in "$M1:5" "$M2:4" "$M3:4"; do
    MID="${MID_RATING%%:*}"
    RATING="${MID_RATING##*:}"
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/feedback" \
        -H "Authorization: Bearer $TOKEN" \
        -H "Content-Type: application/json" \
        -d "{\"movie_id\":$MID,\"rating\":$RATING}")
    [ "$STATUS" = "200" ] || [ "$STATUS" = "201" ] || fail "feedback for movie $MID returned $STATUS"
done
pass "rated 3 movies (5, 4, 4)"

# 7. /recommend now returns kind=personalized and excludes rated movies
echo "=== 7. /recommend returns personalized after 3 ratings ==="
REC_RESP=$(curl -sf "$BASE_URL/recommend" -H "Authorization: Bearer $TOKEN")
KIND=$(echo "$REC_RESP" | jq -r '.kind')
[ "$KIND" = "personalized" ] || fail "/recommend returned kind=$KIND (expected personalized after 3 ratings)"
pass "/recommend returned kind=personalized"

REC_IDS=$(echo "$REC_RESP" | jq '[.movies[] | .movie_id]')
for MID in $M1 $M2 $M3; do
    FOUND=$(echo "$REC_IDS" | jq "index($MID) != null")
    [ "$FOUND" = "false" ] || fail "rated movie $MID appears in /recommend results"
done
pass "/recommend excludes rated movies"

# 8. /because returns at least one row
echo "=== 8. /because returns >=1 row ==="
BECAUSE_COUNT=$(curl -sf "$BASE_URL/because" -H "Authorization: Bearer $TOKEN" | jq '. | length')
[ "$BECAUSE_COUNT" -ge 1 ] || fail "/because returned $BECAUSE_COUNT rows"
pass "/because returned $BECAUSE_COUNT rows"

# 9. Logout and login again
echo "=== 9. Logout and re-login ==="
curl -sf -X POST "$BASE_URL/logout" -H "Authorization: Bearer $TOKEN" > /dev/null

NEW_TOKEN=$(curl -sf -X POST "$BASE_URL/login" \
    -H "Content-Type: application/json" \
    -d "{\"username\":\"$USERNAME\",\"password\":\"$PASSWORD\"}" \
    | jq -r '.token')
[ -n "$NEW_TOKEN" ] && [ "$NEW_TOKEN" != "null" ] || fail "re-login did not return a token"

ME=$(curl -sf "$BASE_URL/me" -H "Authorization: Bearer $NEW_TOKEN" | jq -r '.user_id')
[ -n "$ME" ] || fail "/me failed after re-login"
pass "logout and re-login succeeded, /me works"

# 10. X-Served-By header is present
echo "=== 10. X-Served-By header ==="
SERVED_BY=$(curl -si "$BASE_URL/health" | grep -i "x-served-by" | awk '{print $2}' | tr -d '\r')
[ -n "$SERVED_BY" ] || fail "X-Served-By header missing from /health response"
pass "X-Served-By header present: $SERVED_BY"

echo ""
echo "=== Smoke test PASSED: $PASS checks ==="

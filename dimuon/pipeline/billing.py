"""Print Tinker billing usage for the last N days (default 3), summed by day / event type / model. Read-only."""
import sys, datetime, json, collections
from tinker.lib.public_interfaces.admin_client import AdminClient
n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
now = datetime.datetime.now(datetime.timezone.utc).replace(minute=0, second=0, microsecond=0)
r = AdminClient().get_billing_usage(now - datetime.timedelta(days=n), now + datetime.timedelta(hours=1)).result()
rows = json.loads(r.model_dump_json())["data"]
t = collections.defaultdict(lambda: [0.0, 0])
for x in rows:
    k = (x["bucket_end"][:10], (x.get("event_info") or {}).get("type"), x.get("base_model"))
    t[k][0] += x.get("estimated_cost_usd") or 0
    t[k][1] += x.get("estimated_cost_usd") is None
for k in sorted(t, key=str):
    print(k, "usd %.4f" % t[k][0], "unpriced buckets %d" % t[k][1])
print("last bucket ends", rows[-1]["bucket_end"] if rows else None)

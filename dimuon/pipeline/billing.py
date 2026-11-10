"""Print Tinker billing usage for the last N days (default 3). Read-only."""
import sys, datetime
from tinker.lib.public_interfaces.admin_client import AdminClient
n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
now = datetime.datetime.now(datetime.timezone.utc).replace(minute=0, second=0, microsecond=0)
r = AdminClient().get_billing_usage(now - datetime.timedelta(days=n), now + datetime.timedelta(hours=1)).result()
print(r.model_dump_json(indent=1)[:6000])

# 3. API keys: random, shown once, stored as SHA-256

**Status:** accepted

**Context.** Keys must not be recoverable from a DB leak, and lookup is on the hot path of every request.

**Decision.** Keys are `sk-gw-` + 256 bits from `secrets.token_urlsafe`. Only the SHA-256 digest (and a 10-char prefix for identification) is stored. A slow password hash (bcrypt/argon2) would add tens of ms per request and buys nothing for 256-bit random secrets, which cannot be brute-forced. A 30 s in-process cache removes the DB read.

**Consequences.** Revoking a key (`is_active=false`) takes up to 30 s to take effect per process. Lost keys cannot be recovered, only rotated.

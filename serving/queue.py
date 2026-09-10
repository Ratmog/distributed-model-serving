"""Atomic admission, expiring leases, and token-fenced completion (Redis standalone).

Active records never expire. Completed records expire after result_ttl.
Recovery is at-least-once: a crashed inference may execute again.
"""

import hashlib
import json
import time
import uuid

SUBMIT = """
local old = redis.call('GET', KEYS[1])
if old then
 local split = string.find(old, ':')
 if string.sub(old, 1, split-1) ~= ARGV[1] then return {'conflict'} end
 return {'exists', string.sub(old, split+1)}
end
if tonumber(redis.call('GET', KEYS[2]) or '0') >= tonumber(ARGV[4]) then return {'full'} end
redis.call('SET', KEYS[1], ARGV[1] .. ':' .. ARGV[2])
redis.call('HSET', KEYS[3], 'status', 'queued', 'payload', ARGV[3], 'attempts', 0, 'idem', KEYS[1])
redis.call('RPUSH', KEYS[4], ARGV[2])
redis.call('INCR', KEYS[2])
return {'created', ARGV[2]}
"""
CLAIM = """
local id = redis.call('LPOP', KEYS[1])
if not id then return {} end
local job = ARGV[1] .. id
redis.call('HSET', job, 'status', 'running', 'token', ARGV[2])
redis.call('HINCRBY', job, 'attempts', 1)
redis.call('ZADD', KEYS[2], ARGV[3], id)
return {id, redis.call('HGET', job, 'payload')}
"""
FINISH = """
if redis.call('HGET', KEYS[1], 'token') ~= ARGV[1] then return 0 end
if redis.call('HGET', KEYS[1], 'status') ~= 'running' then return 0 end
redis.call('HSET', KEYS[1], 'status', ARGV[2], 'result', ARGV[3])
redis.call('HDEL', KEYS[1], 'payload', 'token')
redis.call('EXPIRE', KEYS[1], ARGV[4])
local idem = redis.call('HGET', KEYS[1], 'idem')
if idem then redis.call('EXPIRE', idem, ARGV[4]) end
redis.call('ZREM', KEYS[2], ARGV[5])
redis.call('DECR', KEYS[3])
return 1
"""
RECOVER = """
local expired = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1], 'LIMIT', 0, 100)
for _, id in ipairs(expired) do
 local job = ARGV[2] .. id
 redis.call('ZREM', KEYS[1], id)
 redis.call('HDEL', job, 'token')
 if tonumber(redis.call('HGET', job, 'attempts') or '0') >= tonumber(ARGV[3]) then
  redis.call('HSET', job, 'status', 'failed', 'result', '{"error":"Retry budget exhausted"}')
  redis.call('HDEL', job, 'payload')
  redis.call('EXPIRE', job, ARGV[4])
  local idem = redis.call('HGET', job, 'idem')
  if idem then redis.call('EXPIRE', idem, ARGV[4]) end
  redis.call('DECR', KEYS[3])
 else
  redis.call('HSET', job, 'status', 'queued')
  redis.call('RPUSH', KEYS[2], id)
 end
end
return #expired
"""


class QueueFull(Exception):
    pass


class IdempotencyConflict(Exception):
    pass


class JobQueue:
    prefix = "dms:job:"
    waiting = "dms:waiting"
    leases = "dms:leases"
    active = "dms:active"

    def __init__(self, redis, settings):
        self.redis, self.settings = redis, settings

    async def submit(self, request, key=None):
        payload = request.model_dump_json()
        digest = hashlib.sha256(payload.encode()).hexdigest()
        jid = uuid.uuid4().hex
        idem = "dms:idem:" + hashlib.sha256((key or jid).encode()).hexdigest()
        result = await self.redis.eval(
            SUBMIT,
            4,
            idem,
            self.active,
            self.prefix + jid,
            self.waiting,
            digest,
            jid,
            payload,
            self.settings.queue_capacity,
        )
        if result[0] == "full":
            raise QueueFull()
        if result[0] == "conflict":
            raise IdempotencyConflict()
        return result[1]

    async def claim(self, now=None):
        token = uuid.uuid4().hex
        result = await self.redis.eval(
            CLAIM,
            2,
            self.waiting,
            self.leases,
            self.prefix,
            token,
            (time.time() if now is None else now) + self.settings.lease_seconds,
        )
        return (result[0], token, json.loads(result[1])) if result else None

    async def finish(self, jid, token, result, status="succeeded"):
        return await self.redis.eval(
            FINISH,
            3,
            self.prefix + jid,
            self.leases,
            self.active,
            token,
            status,
            json.dumps(result),
            self.settings.result_ttl,
            jid,
        )

    async def recover(self, now=None):
        return await self.redis.eval(
            RECOVER,
            3,
            self.leases,
            self.waiting,
            self.active,
            time.time() if now is None else now,
            self.prefix,
            self.settings.max_attempts,
            self.settings.result_ttl,
        )

    async def get(self, jid):
        record = await self.redis.hgetall(self.prefix + jid)
        if not record:
            return None
        return {
            "id": jid,
            "status": record["status"],
            "attempts": int(record["attempts"]),
            "result": json.loads(record["result"]) if "result" in record else None,
        }

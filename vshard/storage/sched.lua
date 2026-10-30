--
-- Scheduler module ensures fair time sharing between incompatible operations:
-- storage refs and bucket moves.
-- Storage ref is supposed to prevent all bucket moves and provide safe
-- environment for all kinds of possible requests on entire dataset of all
-- spaces stored on the instance.
-- Bucket move, on the contrary, wants to make a part of the dataset not usable
-- temporary.
-- Without a scheduler it would be possible to always keep at least one ref on
-- the storage and block bucket moves forever. Or vice versa - during
-- rebalancing block all incoming refs for the entire time of data migration,
-- essentially making map-reduce not usable since it heavily depends on refs.
--
-- The schedule divides storage time between refs and moves so both of them can
-- execute without blocking each other. Division proportions depend on the
-- configuration settings.
--
-- Idea of non-blockage is based on quotas and strikes. Move and ref both have
-- quotas. When one op executes more than quota requests in a row (makes a
-- strike) while the other op has queued requests, the first op stops accepting
-- new requests until the other op executes.
--

local MODULE_INTERNALS = '__module_vshard_storage_sched'
-- Update when change behaviour of anything in the file, to be able to reload.
local MODULE_VERSION = 1

local ffi = require('ffi')
local lfiber = require('fiber')
local lconsts = require('vshard.consts')
local lregistry = require('vshard.registry')
local lutil = require('vshard.util')
local fiber_clock = lfiber.clock
local fiber_cond_wait = lutil.fiber_cond_wait
local fiber_is_self_canceled = lutil.fiber_is_self_canceled

--
-- The scheduler numbers live in a C struct instead of a Lua table. A host
-- process can then take and release a ref without calling Lua: Picodata does
-- it when no move is running or queued. The host owns the struct and exports
-- two functions to get it, see vshard_sched_state in Picodata. Keep the layout
-- in sync with the host: the size is checked here, the field offsets by the
-- host's tests.
--
-- Doubles, because LuaJIT reads them as plain Lua numbers, while int64_t would
-- be boxed into a cdata on every read. The quotas are not always integers.
--
-- A second ffi.cdef of the same struct fails, and the module can be loaded
-- again after its package.loaded entry is cleared.
--
if not pcall(ffi.typeof, 'struct vshard_sched_state') then
    ffi.cdef([[
        struct vshard_sched_state {
            /* Refs. */
            /* Number of ref requests being executed. */
            double ref_count;
            /* Number of ref requests waiting for start. */
            double ref_queue;
            /*
             * Number of ref requests executed in a row. When becomes bigger
             * than quota, any next queued move blocks new refs.
             */
            double ref_strike;
            double ref_quota;
            /* Moves. */
            /* Number of move requests being executed. */
            double move_count;
            /* Number of move requests waiting for start. */
            double move_queue;
            /*
             * Number of move requests executed in a row. When becomes bigger
             * than quota, any next queued ref blocks new moves.
             */
            double move_strike;
            double move_quota;
            /*
             * Whether all the buckets of the storage are writable: 1 yes,
             * 0 no, -1 unknown. Written by the storage module, which caches
             * the same value.
             */
            double all_rw;
        };
        struct vshard_sched_state *pico_vshard_sched_state(void);
        size_t pico_vshard_sched_state_size(void);
    ]])
end

local function sched_state_new()
    local ok, state = pcall(function() return ffi.C.pico_vshard_sched_state() end)
    if not ok then
        -- No host to share the state with.
        state = ffi.new('struct vshard_sched_state')
        state.all_rw = -1
        return state
    end
    local size = tonumber(ffi.C.pico_vshard_sched_state_size())
    if size ~= ffi.sizeof('struct vshard_sched_state') then
        error(string.format('struct vshard_sched_state is %d bytes in the '..
                            'host and %d bytes in vshard', size,
                            ffi.sizeof('struct vshard_sched_state')))
    end
    return state
end

local S = sched_state_new()
S.ref_quota = lconsts.DEFAULT_SCHED_REF_QUOTA
S.move_quota = lconsts.DEFAULT_SCHED_MOVE_QUOTA

local M = rawget(_G, MODULE_INTERNALS)
if not M then
    M = {
        ---------------- Common module attributes ----------------
        module_version = MODULE_VERSION,
        -- Scheduler condition is signaled every time anything significant
        -- happens - count of an operation type drops to 0, or quota increased,
        -- etc.
        cond = lfiber.cond(),
        -- Refs and moves counters, see struct vshard_sched_state.
        state = S,
    }
else
    return M
end

local function sched_wait_anything(timeout)
    return fiber_cond_wait(M.cond, timeout)
end

--
-- Return the remaining timeout in case there was a yield. This helps to save
-- current clock get in the caller code if there were no yields.
--
local function sched_ref_start(timeout)
    local deadline, ok, err
    -- Fast-path. Moves are extremely rare. No need to inc-dec the ref queue
    -- then nor try to start some loops.
    if S.move_count == 0 and S.move_queue == 0 then
        goto success
    end
    deadline = fiber_clock() + timeout

    S.ref_queue = S.ref_queue + 1

::retry::
    if S.move_count > 0 then
        goto wait_and_retry
    end
    -- Even if move count is zero, must ensure the time usage is fair. Does not
    -- matter in case the moves have no quota at all. That allows to ignore them
    -- infinitely until all refs end voluntarily.
    if S.move_queue > 0 and S.ref_strike >= S.ref_quota and
       S.move_quota > 0 then
        goto wait_and_retry
    end

    S.ref_queue = S.ref_queue - 1

::success::
    S.ref_count = S.ref_count + 1
    S.ref_strike = S.ref_strike + 1
    S.move_strike = 0
    do return timeout end

::wait_and_retry::
    ok, err = sched_wait_anything(timeout)
    if not ok then
        S.ref_queue = S.ref_queue - 1
        return nil, err
    end
    timeout = deadline - fiber_clock()
    goto retry
end

local function sched_ref_end(count)
    count = S.ref_count - count
    S.ref_count = count
    if count == 0 and S.move_queue > 0 then
        M.cond:broadcast()
    end
end

--
-- Return the remaining timeout in case there was a yield. This helps to save
-- current clock get in the caller code if there were no yields.
--
local function sched_move_start(timeout)
    local ok, err, deadline, ref_deadline
    local lref = lregistry.storage_ref
    -- Fast-path. Refs are not extremely rare *when used*. But they are not
    -- expected to be used in a lot of installations. So most of the times the
    -- moves should work right away.
    if S.ref_count == 0 and S.ref_queue == 0 then
        goto success
    end
    deadline = fiber_clock() + timeout

    S.move_queue = S.move_queue + 1

::retry::
    if S.ref_count > 0 then
        ref_deadline = lref.next_deadline()
        if ref_deadline < deadline then
            timeout = ref_deadline - fiber_clock()
        end
        ok, err = sched_wait_anything(timeout)
        timeout = deadline - fiber_clock()
        if ok then
            goto retry
        end
        if fiber_is_self_canceled() then
            goto fail
        end
        -- Even if the timeout has expired already (or was 0 from the
        -- beginning), it is still possible the move can be started if all the
        -- present refs are expired too and can be collected.
        lref.gc()
        -- GC could yield - need to refetch the clock again.
        timeout = deadline - fiber_clock()
        if S.ref_count > 0 then
            if timeout < 0 then
                goto fail
            end
            goto retry
        end
    end

    if S.ref_queue > 0 and S.move_strike >= S.move_quota and
       S.ref_quota > 0 then
        ok, err = sched_wait_anything(timeout)
        if not ok then
            goto fail
        end
        timeout = deadline - fiber_clock()
        goto retry
    end

    S.move_queue = S.move_queue - 1

::success::
    S.move_count = S.move_count + 1
    S.move_strike = S.move_strike + 1
    S.ref_strike = 0
    do return timeout end

::fail::
    S.move_queue = S.move_queue - 1
    return nil, err
end

local function sched_move_end(count)
    count = S.move_count - count
    S.move_count = count
    if count == 0 and S.ref_queue > 0 then
        M.cond:broadcast()
    end
end

local function sched_cfg(cfg)
    local new_ref_quota = cfg.sched_ref_quota
    local new_move_quota = cfg.sched_move_quota

    if new_ref_quota then
        if new_ref_quota > S.ref_quota then
            M.cond:broadcast()
        end
        S.ref_quota = new_ref_quota
    end
    if new_move_quota then
        if new_move_quota > S.move_quota then
            M.cond:broadcast()
        end
        S.move_quota = new_move_quota
    end
end

M.ref_start = sched_ref_start
M.ref_end = sched_ref_end
M.move_start = sched_move_start
M.move_end = sched_move_end
M.cfg = sched_cfg

--
-- The numbers used to be the fields of M. Tests and other code still read and
-- write them there, so M forwards these names to the struct.
--
local state_fields = {
    ref_count = true, ref_queue = true, ref_strike = true, ref_quota = true,
    move_count = true, move_queue = true, move_strike = true, move_quota = true,
    all_rw = true,
}
setmetatable(M, {
    __index = function(_, key)
        if state_fields[key] then
            return S[key]
        end
    end,
    __newindex = function(self, key, value)
        if state_fields[key] then
            S[key] = value
        else
            rawset(self, key, value)
        end
    end,
})

lregistry.storage_sched = M

return M

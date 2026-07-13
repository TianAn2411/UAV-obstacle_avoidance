import numpy as np
import subprocess
import tempfile
import os
import time

try:
    from utils.gz_transport_client import GzTransportClient
except Exception:
    GzTransportClient = None


def _log_gz_transport_fallback(message):
    if os.environ.get("GZ_TRANSPORT_FALLBACK_LOG", "1") == "1":
        print(message, flush=True)


def _client_from_env(env=None):
    if GzTransportClient is None:
        return None
    gz_partition = None
    if env:
        gz_partition = env.get("GZ_PARTITION")
    return GzTransportClient(gz_partition=gz_partition, use_lock=True)

def poisson_disk(n, region, min_dist, max_tries=5000, exclusion_zones=None):
    if exclusion_zones is None:
        exclusion_zones = []

    xmin, ymin, xmax, ymax = region
    pts = []
    tries = 0
    while len(pts) < n and tries < max_tries:
        p = np.array([np.random.uniform(xmin, xmax),
                      np.random.uniform(ymin, ymax)])

        # Check distance to existing pillars
        valid = True
        for q in pts:
            if np.linalg.norm(p - q) < min_dist:
                valid = False
                break

        # Check exclusion zones
        if valid:
            for ex_x, ex_y, ex_r in exclusion_zones:
                if np.linalg.norm(p - np.array([ex_x, ex_y])) < ex_r:
                    valid = False
                    break

        if valid:
            pts.append(p)
        tries += 1

    if len(pts) < n:
        raise RuntimeError(f"poisson_disk failed to spawn {n} pillars after {max_tries} tries. Region: {region}, Exclusions: {exclusion_zones}")

    return np.array(pts)

def make_disc_marker_sdf(name, x, y, r, g, b, radius=0.5, height=0.08):
    return f"""
<sdf version='1.9'>
    <model name='{name}'>
    <static>true</static>
    <pose>{x} {y} {height/2} 0 0 0</pose>
    <link name='link'>
<visual name='v'>
<geometry><cylinder>
<radius>{radius}</radius><length>{height}</length>
</cylinder></geometry>
<material><diffuse>{r} {g} {b} 1</diffuse><ambient>{r*0.5} {g*0.5} {b*0.5} 1</ambient></material>
</visual>
</link>
</model>
</sdf>"""


def spawn_disc_marker(name, x, y, r, g, b, radius=0.5, world_name="default", env=None):
    sdf_raw = make_disc_marker_sdf(name, x, y, r, g, b, radius=radius)
    client = _client_from_env(env)
    if client is not None and client.available():
        try:
            ok = client.create_sdf(world_name=world_name, name=name, sdf=sdf_raw, timeout_ms=3000)
            if ok:
                return True
        except Exception:
            pass
    # CLI fallback
    sdf_escaped = sdf_raw.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
    try:
        import subprocess
        env_vars = dict(os.environ)
        if env and env.get("GZ_PARTITION"):
            env_vars["GZ_PARTITION"] = env["GZ_PARTITION"]
        subprocess.run(
            ["gz", "service", "-s", f"/world/{world_name}/create",
             "--reqtype", "gz.msgs.EntityFactory",
             "--reptype", "gz.msgs.Boolean",
             "--timeout", "3000",
             "--req", f'name: "{name}" sdf: "{sdf_escaped}"'],
            timeout=5.0, capture_output=True, env=env_vars,
        )
    except Exception:
        pass
    return False


def make_pillar_sdf(name, x, y, radius, height):
    return f"""
<sdf version='1.9'>
    <model name='{name}'>
    <static>true</static>
    <pose>{x} {y} {height/2} 0 0 0</pose>
    <link name='link'>
<visual name='v'>
<geometry><cylinder>
<radius>{radius}</radius><length>{height}</length>
</cylinder></geometry>
<material><diffuse>0.5 0.3 0.2 1</diffuse></material>
</visual>
</link>
</model>
</sdf>"""

def spawn_pillar(name, x, y, radius=0.3, height=6.0, world_name="default", env=None):
    sdf_raw = make_pillar_sdf(name, x, y, radius, height)
    client = _client_from_env(env)
    if client is not None and client.available():
        try:
            ok = client.create_sdf(
                world_name=world_name,
                name=name,
                sdf=sdf_raw,
                timeout_ms=3000,
            )
            if ok:
                return True
            _log_gz_transport_fallback(
                f"[GZ TRANSPORT] create_sdf failed; fallback CLI world={world_name} name={name}"
            )
        except Exception as exc:
            _log_gz_transport_fallback(
                f"[GZ TRANSPORT] create_sdf failed; fallback CLI world={world_name} name={name} error={exc}"
            )

    # Escape for protobuf text format: quotes and newlines
    sdf_escaped = sdf_raw.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')

    cmd = ["gz", "service", "-s", f"/world/{world_name}/create",
           "--reqtype", "gz.msgs.EntityFactory",
           "--reptype", "gz.msgs.Boolean",
           "--timeout", "5000",
           "--req", f'sdf: "{sdf_escaped}" name: "{name}" allow_renaming: false']

    max_retries = 3
    for attempt in range(max_retries):
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        if result.returncode == 0 and "data: true" in result.stdout.lower():
            return True  # Success

        if "timed out" in result.stderr.lower() and attempt < max_retries - 1:
            time.sleep(0.5)
            continue

        raise RuntimeError(
            f"Gazebo service failed to spawn {name} (attempt {attempt+1}/{max_retries}). "
            f"returncode={result.returncode}, stdout={result.stdout}, stderr={result.stderr}"
        )


def make_dynamic_pillar_sdf(name, x, y, radius, height, mass=5.0):
    """Same visual cylinder as make_pillar_sdf but a non-static body (gravity
    off, no collision -- same as the static pillar, which has none either) so
    the VelocityControl system plugin can drive it continuously in Gazebo's
    own physics loop instead of us teleporting it via set_pose every step."""
    izz = 0.5 * mass * radius * radius
    ixx = iyy = (mass / 12.0) * (3.0 * radius * radius + height * height)
    return f"""
<sdf version='1.9'>
    <model name='{name}'>
    <static>false</static>
    <pose>{x} {y} {height/2} 0 0 0</pose>
    <link name='link'>
    <gravity>false</gravity>
    <inertial>
      <mass>{mass}</mass>
      <inertia>
        <ixx>{ixx}</ixx><iyy>{iyy}</iyy><izz>{izz}</izz>
        <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
      </inertia>
    </inertial>
<visual name='v'>
<geometry><cylinder>
<radius>{radius}</radius><length>{height}</length>
</cylinder></geometry>
<material><diffuse>0.5 0.3 0.2 1</diffuse></material>
</visual>
</link>
<plugin filename="gz-sim-velocity-control-system" name="gz::sim::systems::VelocityControl">
</plugin>
</model>
</sdf>"""


def spawn_dynamic_pillar(name, x, y, radius=0.3, height=6.0, mass=5.0, world_name="default", env=None):
    sdf_raw = make_dynamic_pillar_sdf(name, x, y, radius, height, mass=mass)
    client = _client_from_env(env)
    if client is not None and client.available():
        try:
            ok = client.create_sdf(
                world_name=world_name,
                name=name,
                sdf=sdf_raw,
                timeout_ms=3000,
            )
            if ok:
                return True
            _log_gz_transport_fallback(
                f"[GZ TRANSPORT] create_sdf failed; fallback CLI world={world_name} name={name}"
            )
        except Exception as exc:
            _log_gz_transport_fallback(
                f"[GZ TRANSPORT] create_sdf failed; fallback CLI world={world_name} name={name} error={exc}"
            )

    sdf_escaped = sdf_raw.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')

    cmd = ["gz", "service", "-s", f"/world/{world_name}/create",
           "--reqtype", "gz.msgs.EntityFactory",
           "--reptype", "gz.msgs.Boolean",
           "--timeout", "5000",
           "--req", f'sdf: "{sdf_escaped}" name: "{name}" allow_renaming: false']

    max_retries = 3
    for attempt in range(max_retries):
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        if result.returncode == 0 and "data: true" in result.stdout.lower():
            return True

        if "timed out" in result.stderr.lower() and attempt < max_retries - 1:
            time.sleep(0.5)
            continue

        raise RuntimeError(
            f"Gazebo service failed to spawn {name} (attempt {attempt+1}/{max_retries}). "
            f"returncode={result.returncode}, stdout={result.stdout}, stderr={result.stderr}"
        )


def set_pillar_velocity(name, vx, vy, vz=0.0, env=None, timeout_ms=1000):
    """Publish a Twist to /model/<name>/cmd_vel -- VelocityControl system
    plugin (added by make_dynamic_pillar_sdf) applies it every physics tick
    until overwritten, so this is a fire-once-per-heading-change call, not a
    per-RL-step one."""
    client = _client_from_env(env)
    if client is not None and client.available():
        try:
            if client.set_model_velocity(name, vx, vy, vz):
                return True
            _log_gz_transport_fallback(
                f"[GZ TRANSPORT] set_model_velocity failed; fallback CLI name={name}"
            )
        except Exception as exc:
            _log_gz_transport_fallback(
                f"[GZ TRANSPORT] set_model_velocity failed; fallback CLI name={name} error={exc}"
            )

    cmd = [
        "gz", "topic", "-t", f"/model/{name}/cmd_vel",
        "-m", "gz.msgs.Twist",
        "-p", f"linear: {{x: {float(vx)}, y: {float(vy)}, z: {float(vz)}}}",
    ]
    try:
        result = subprocess.run(cmd, timeout=timeout_ms / 1000.0, capture_output=True, env=env)
        return result.returncode == 0
    except Exception:
        return False


def move_entity(name, x, y, z, world_name="default", env=None, timeout_ms=3000):
    client = _client_from_env(env)
    if client is not None and client.available():
        try:
            ok = client.set_pose(
                world_name=world_name,
                name=name,
                x=x,
                y=y,
                z=z,
                yaw=0.0,
                timeout_ms=timeout_ms,
            )
            if ok:
                return True
            _log_gz_transport_fallback(
                f"[GZ TRANSPORT] set_pose failed; fallback CLI world={world_name} name={name}"
            )
        except Exception as exc:
            _log_gz_transport_fallback(
                f"[GZ TRANSPORT] set_pose failed; fallback CLI world={world_name} name={name} error={exc}"
            )

    cmd = [
        "gz",
        "service",
        "-s",
        f"/world/{world_name}/set_pose",
        "--reqtype",
        "gz.msgs.Pose",
        "--reptype",
        "gz.msgs.Boolean",
        "--timeout",
        str(timeout_ms),
        "--req",
        (
            f'name: "{name}" '
            f"position {{ x: {float(x)} y: {float(y)} z: {float(z)} }} "
            "orientation { x: 0 y: 0 z: 0 w: 1 }"
        ),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode == 0 and "data: true" in (result.stdout or "").lower():
        return True

    raise RuntimeError(
        f"Gazebo service failed to move {name}. "
        f"returncode={result.returncode}, stdout={result.stdout}, stderr={result.stderr}"
    )

def _pose_vector_block(name: str, x: float, y: float, z: float, yaw: float = 0.0) -> str:
    import math

    qz = math.sin(float(yaw) * 0.5)
    qw = math.cos(float(yaw) * 0.5)

    return (
        "pose { "
        f'name: "{name}" '
        f"position {{ x: {float(x):.6f} y: {float(y):.6f} z: {float(z):.6f} }} "
        f"orientation {{ x: 0 y: 0 z: {qz:.8f} w: {qw:.8f} }} "
        "}"
    )


def move_entities_batch(poses, world_name="default", env=None, timeout_ms=None):
    """Move nhiều entity trong một Gazebo service call bằng /set_pose_vector.

    Args:
        poses: list[dict] với keys: name, x, y, z, yaw (optional)
               hoặc list[tuple]: (name, x, y, z) hay (name, x, y, z, yaw)
        world_name: Gazebo world name.
        env: Environment variables với GZ_PARTITION đúng env.
        timeout_ms: Timeout cho service call.

    Returns:
        True nếu Gazebo trả data: true.

    Raises:
        RuntimeError nếu service fail sau retry.
    """
    if not poses:
        return True
    if timeout_ms is None:
        timeout_ms = int(os.environ.get("GZ_SET_POSE_VECTOR_TIMEOUT_MS", "2500"))

    client = _client_from_env(env)
    if client is not None and client.available():
        try:
            ok = client.set_pose_vector(
                world_name=world_name,
                poses=poses,
                timeout_ms=timeout_ms,
            )
            if ok:
                return True
            _log_gz_transport_fallback(
                f"[GZ TRANSPORT] set_pose_vector failed; fallback CLI world={world_name} count={len(poses)} timeout_ms={timeout_ms}"
            )
        except Exception as exc:
            _log_gz_transport_fallback(
                f"[GZ TRANSPORT] set_pose_vector failed; fallback CLI world={world_name} count={len(poses)} timeout_ms={timeout_ms} error={exc}"
            )

    blocks = []
    for item in poses:
        if isinstance(item, dict):
            name = item["name"]
            x, y, z = item["x"], item["y"], item["z"]
            yaw = item.get("yaw", 0.0)
        elif len(item) == 4:
            name, x, y, z = item
            yaw = 0.0
        elif len(item) == 5:
            name, x, y, z, yaw = item
        else:
            raise ValueError(f"Invalid pose tuple: {item}")
        blocks.append(_pose_vector_block(name, x, y, z, yaw))

    req = "\n".join(blocks)

    cmd = [
        "gz", "service",
        "-s", f"/world/{world_name}/set_pose_vector/blocking",
        "--reqtype", "gz.msgs.Pose_V",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", str(int(timeout_ms)),
        "--req", req,
    ]

    max_retries = 3
    last_result = None
    for attempt in range(1, max_retries + 1):
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        last_result = result
        if result.returncode == 0 and "data: true" in (result.stdout or "").lower():
            return True
        if attempt < max_retries:
            time.sleep(0.25 * attempt)

    raise RuntimeError(
        "Gazebo service failed to batch move entities. "
        f"count={len(poses)}, "
        f"returncode={last_result.returncode if last_result else 'none'}, "
        f"stdout={last_result.stdout if last_result else ''}, "
        f"stderr={last_result.stderr if last_result else ''}"
    )


def clear_pillars(num_to_check=50, world_name="default", name_prefix="pillar", env=None):
    client = _client_from_env(env)
    fallback_logged = False
    for i in range(num_to_check):
        pillar_name = f"{name_prefix}_{i}"
        if client is not None and client.available():
            try:
                if client.remove_model(world_name=world_name, name=pillar_name, timeout_ms=2000):
                    continue
                if not fallback_logged:
                    _log_gz_transport_fallback(
                        f"[GZ TRANSPORT] remove_model fallback CLI enabled world={world_name} prefix={name_prefix}"
                    )
                    fallback_logged = True
            except Exception:
                if not fallback_logged:
                    _log_gz_transport_fallback(
                        f"[GZ TRANSPORT] remove_model fallback CLI enabled world={world_name} prefix={name_prefix}"
                    )
                    fallback_logged = True

        cmd = ["gz", "service", "-s", f"/world/{world_name}/remove",
               "--reqtype", "gz.msgs.Entity",
               "--reptype", "gz.msgs.Boolean",
               "--timeout", "2000",
               "--req", f'name: "{pillar_name}" type: 2']
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)

def sample_corridor_pillars(
    start,
    goal,
    num_main,
    num_decor=0,
    corridor_half_width=4.5,
    min_pillar_dist=1.6,
    max_pillar_dist=2.0,
    start_clearance=2.5,
    goal_clearance=2.5,
    t_min=0.25,
    t_max=0.85,
    spawn_bounds=(-10.0, -9.0, 10.0, 9.0),
    corridor_jitter_deg=0.0,
    decor_lateral_max=None,
    decor_fill_prob=0.35,
    max_tries_per_pillar=300,
):
    """Pure random spawn -- rejection sampling (Poisson-disk-style) inside
    the corridor rectangle, the ONLY placement constraint is pairwise
    min_pillar_dist clearance against every already-placed pillar.

    Replaces the dense jittered-grid design (row/col cumulative-sum, one
    mandatory gap per row): that design's rows shared an exact t-coordinate
    by construction, so every row was a dead-straight wall perpendicular to
    the corridor with exactly one hole -- too regular a pattern, a policy
    could overfit to "find the one gap in the next perpendicular wall"
    instead of learning general obstacle avoidance. This trades that
    design's "always solvable by construction" guarantee for genuine
    randomness (user's explicit call): an episode can now rarely spawn with
    no flyable path through the main field. Not treated as a bug -- if it
    becomes a real training problem, the fix is a post-hoc connectivity
    check + retry, not a return to the structured grid.

    max_pillar_dist is accepted for call-site compatibility (the fallback
    ladder in pillar_manager.py still loosens it) but unused here -- pure
    random rejection sampling only has a clearance FLOOR, no upper bound
    on spacing.

    Decor pillars use the same rejection-sampling loop, scattered outside
    +-corridor_half_width out to decor_lateral_max with a flat
    decor_fill_prob -- they only exist to punish a wide detour around the
    whole main field, never need to be threadable themselves.
    """
    start_xy = np.array(start[:2])
    goal_xy = np.array(goal[:2])

    direction = goal_xy - start_xy
    dist = np.linalg.norm(direction)
    if dist > 0:
        direction = direction / dist
    else:
        direction = np.array([1.0, 0.0])

    if corridor_jitter_deg > 0.0:
        jitter_rad = np.random.uniform(
            -abs(float(corridor_jitter_deg)) * np.pi / 180.0,
            abs(float(corridor_jitter_deg)) * np.pi / 180.0,
        )
        cos_j, sin_j = np.cos(jitter_rad), np.sin(jitter_rad)
        direction = np.array([
            cos_j * direction[0] - sin_j * direction[1],
            sin_j * direction[0] + cos_j * direction[1],
        ])

    normal = np.array([-direction[1], direction[0]])
    xmin, ymin, xmax, ymax = spawn_bounds

    num_main = int(num_main)
    num_decor = int(num_decor)
    total = num_main + num_decor
    if total <= 0:
        return []

    decor_lateral_max = float(decor_lateral_max) if decor_lateral_max is not None else corridor_half_width
    min_pillar_dist = float(min_pillar_dist)

    def _valid_xy(p):
        if not (xmin <= p[0] <= xmax and ymin <= p[1] <= ymax):
            return False
        if np.linalg.norm(p - start_xy) < start_clearance:
            return False
        if np.linalg.norm(p - goal_xy) < goal_clearance:
            return False
        return True

    placed: list[dict] = []  # {"xy", "t", "lateral", "blocking", "formation_id"}

    def _try_place(t_lo, t_hi, lat_lo, lat_hi, blocking):
        for _ in range(max_tries_per_pillar):
            t_frac = float(np.random.uniform(t_lo, t_hi))
            lat = float(np.random.uniform(lat_lo, lat_hi))
            p = start_xy + (t_frac * dist) * direction + lat * normal
            if not _valid_xy(p):
                continue
            if any(np.linalg.norm(p - prev["xy"]) < min_pillar_dist for prev in placed):
                continue
            placed.append({
                "xy": p, "t": t_frac, "lateral": lat,
                "blocking": blocking, "formation_id": None,
            })
            return True
        return False

    # ---- Main field: pure random inside +-corridor_half_width ----
    main_placed = 0
    for _ in range(num_main):
        if _try_place(t_min, t_max, -corridor_half_width, corridor_half_width, True):
            main_placed += 1
        else:
            break  # region saturated at this clearance -- further tries won't help
    if main_placed < num_main:
        print(f"Warning: random spawn only fit {main_placed}/{num_main} main pillars in corridor (min_dist={min_pillar_dist}).")

    # ---- Decor field: pure random outside +-corridor_half_width ----
    decor_placed = 0
    for _ in range(num_decor):
        if np.random.random() >= decor_fill_prob:
            continue
        if np.random.random() < 0.5:
            lat_lo, lat_hi = corridor_half_width, decor_lateral_max
        else:
            lat_lo, lat_hi = -decor_lateral_max, -corridor_half_width
        if _try_place(t_min, t_max, lat_lo, lat_hi, False):
            decor_placed += 1

    if len(placed) < total:
        print(f"Warning: Only spawned {len(placed)}/{total} pillars in corridor (random spawn).")

    return placed


def sample_random_field_metadata(
    num_main=0,
    num_decor=0,
    region=(-7, -6, 7, 6),
    min_dist=1.6,
    max_dist=2.0,
    start=None,
    goal=None,
    name_prefix="pillar",
    corridor_half_width=4.5,
    start_clearance=2.5,
    goal_clearance=2.5,
    t_min=0.25,
    t_max=0.85,
    spawn_bounds=(-10.0, -9.0, 10.0, 9.0),
    pillar_radius_range=(0.2, 0.4),
    pillar_height_range=(4.0, 6.0),
    corridor_jitter_deg=0.0,
    decor_lateral_max=None,
    decor_fill_prob=0.35,
):
    total = int(num_main) + int(num_decor)
    if total <= 0:
        return []

    if start is not None and goal is not None:
        pts = sample_corridor_pillars(
            start=start,
            goal=goal,
            num_main=num_main,
            num_decor=num_decor,
            corridor_half_width=corridor_half_width,
            min_pillar_dist=min_dist,
            max_pillar_dist=max_dist,
            start_clearance=start_clearance,
            goal_clearance=goal_clearance,
            t_min=t_min,
            t_max=t_max,
            spawn_bounds=spawn_bounds,
            corridor_jitter_deg=corridor_jitter_deg,
            decor_lateral_max=decor_lateral_max,
            decor_fill_prob=decor_fill_prob,
        )
    else:
        exclusion_zones = []
        if start is not None:
            exclusion_zones.append((start[0], start[1], start_clearance))
        if goal is not None:
            exclusion_zones.append((goal[0], goal[1], goal_clearance))
        pts = poisson_disk(total, region, min_dist, exclusion_zones=exclusion_zones)

    metadata = []
    for i, p in enumerate(pts):
        # sample_corridor_pillars returns dicts (carries the spawn-time
        # blocking/decor flag through to the caller); poisson_disk returns
        # bare xy pairs with no such concept -- treat those as blocking since
        # there's no corridor structure to be "decor" relative to.
        if isinstance(p, dict):
            x, y = float(p["xy"][0]), float(p["xy"][1])
            is_blocking = bool(p.get("blocking", True))
            formation_id = p.get("formation_id")
        else:
            x, y = p[0], p[1]
            is_blocking = True
            formation_id = None
        r = np.random.uniform(pillar_radius_range[0], pillar_radius_range[1])
        h = np.random.uniform(pillar_height_range[0], pillar_height_range[1])
        pname = f"{name_prefix}_{i}"
        metadata.append({
            "name": pname,
            "x": x,
            "y": y,
            "radius": r,
            "height": h,
            "blocking": is_blocking,
            "formation_id": formation_id,
        })
    return metadata


def spawn_random_field(
    num_main=0,
    num_decor=0,
    region=(-7, -6, 7, 6),
    min_dist=1.6,
    max_dist=2.0,
    world_name="default",
    start=None,
    goal=None,
    name_prefix="pillar",
    env=None,
    corridor_half_width=4.5,
    start_clearance=2.5,
    goal_clearance=2.5,
    t_min=0.25,
    t_max=0.85,
    spawn_bounds=(-10.0, -9.0, 10.0, 9.0),
    pillar_radius_range=(0.2, 0.4),
    pillar_height_range=(4.0, 6.0),
    decor_lateral_max=None,
    decor_fill_prob=0.35,
):
    metadata = sample_random_field_metadata(
        num_main=num_main,
        num_decor=num_decor,
        region=region,
        min_dist=min_dist,
        max_dist=max_dist,
        start=start,
        goal=goal,
        name_prefix=name_prefix,
        corridor_half_width=corridor_half_width,
        start_clearance=start_clearance,
        goal_clearance=goal_clearance,
        t_min=t_min,
        t_max=t_max,
        spawn_bounds=spawn_bounds,
        pillar_radius_range=pillar_radius_range,
        pillar_height_range=pillar_height_range,
        decor_lateral_max=decor_lateral_max,
        decor_fill_prob=decor_fill_prob,
    )

    for m in metadata:
        spawn_pillar(
            m["name"],
            m["x"],
            m["y"],
            m["radius"],
            m["height"],
            world_name=world_name,
            env=env,
        )

    return metadata

if __name__ == "__main__":
    clear_pillars()
    spawn_random_field()

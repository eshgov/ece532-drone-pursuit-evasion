import pygame
import math
import random
import numpy as np
import json
import datetime
from typing import Dict, Any, List

###############################################################################
# SHARED RESULT CLASS (NOW INCLUDES BLACK SWAN TRIGGER LOGGING)
###############################################################################

class SimulationResult:
    def __init__(self):
        self.goals_reached: int = 0
        self.caught: bool = False
        self.time_elapsed: float = 0.0
        self.distance_trace: List[float] = []
        self.goal_times: List[float] = []
        self.black_swan_trigger_goal: int | None = None  # NEW FIELD
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'goals_reached': self.goals_reached,
            'caught': self.caught,
            'time_elapsed': self.time_elapsed,
            'distance_trace': self.distance_trace,
            'goal_times': self.goal_times,
            'black_swan_trigger_goal': self.black_swan_trigger_goal
        }

###############################################################################
# BLACK SWAN SIMULATION FUNCTION
###############################################################################

def run_black_swan_filter(seed: int = None, max_goals: int = 10, max_time: float = 30.0) -> Dict[str, Any]:

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    result = SimulationResult()

    ###########################################################################
    # SETTINGS
    ###########################################################################
    w, h = 800, 600
    evader_speed = 8.0

    PURSUER_SPEED_BASE = 1.2
    PURSUER_SPEED_COOP = 2.5

    evader_radius = pursuer_radius = 15
    goal_radius = 20
    COLLISION_DISTANCE = evader_radius + pursuer_radius

    CBF_SAFETY_RADIUS = 80.0
    STRAY_RADIUS = 150.0
    CBF_ALPHA = 2.0
    CBF_EPS = 1e-9
    CBF_MAX_ITERS = 3

    BLOCK_FRACTION = 0.55
    BLOCK_STANDOFF = 40.0
    BLOCK_REPOSITION_GAIN = 1.0
    ANGLE_SEP_MIN = math.radians(80)

    WHITE, BLUE, RED, GREEN = (255,255,255), (50,150,255), (255,80,80), (50,200,50)
    BLACK, PURPLE, ORANGE = (0,0,0), (150,0,150), (255,140,0)
    LIGHT_BLUE, LIGHT_GRAY, TEAL = (200,200,255), (220,220,220), (0,180,180)

    ###########################################################################
    # Initialize pygame
    ###########################################################################
    pygame.init()
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Black Swan – Cooperative CBF Pursuit-Evasion")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    ###########################################################################
    # INITIAL STATE
    ###########################################################################
    evader_x = w // 4
    evader_y = h // 2

    p1_x = 3 * w // 4
    p1_y = h // 2

    p2_x, p2_y = None, None
    p2_active = False

    p1_speed = PURSUER_SPEED_BASE
    p2_speed = PURSUER_SPEED_BASE

    goal_x, goal_y = random.randint(0, w), random.randint(0, h)
    goal_counter = 0
    goal_times = []

    # Random black swan spawn timing
    black_swan_goal_index = random.randint(1, max(1, max_goals - 1))

    prev_evader_x, prev_evader_y = evader_x, evader_y
    prev_p1_x, prev_p1_y = p1_x, p1_y
    prev_p2_x, prev_p2_y = None, None

    cbf_is_active = False
    black_swan_announced = False
    announce_timer = 0

    running = True
    game_over = False
    w_goal, w_evade = 1.0, 0.0

    start_time = pygame.time.get_ticks() / 1000.0

    ###########################################################################
    # HELPERS
    ###########################################################################
    def clamp_speed(vx, vy, vmax):
        mag = math.hypot(vx, vy)
        if mag <= vmax or mag < CBF_EPS:
            return vx, vy
        return vx * vmax / mag, vy * vmax / mag

    def spawn_far_from_evader(min_dist=120):
        for _ in range(20):
            x = random.randint(0, w)
            y = random.randint(0, h)
            if math.hypot(x - evader_x, y - evader_y) >= min_dist:
                return x, y
        return (w - evader_x, h - evader_y)

    def blend_weights(dist):
        if dist >= STRAY_RADIUS:
            return 1.0, 0.0
        if dist <= CBF_SAFETY_RADIUS:
            return 0.0, 1.0
        t = (dist - CBF_SAFETY_RADIUS) / max(STRAY_RADIUS - CBF_SAFETY_RADIUS, 1e-6)
        w_ev = 1.0 - t
        w_go = 1.0 - w_ev
        return w_go, w_ev

    def goal_block_point(ev, goal):
        ex, ey = ev
        gx, gy = goal
        dx, dy = gx - ex, gy - ey
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return gx, gy
        usable = max(dist - BLOCK_STANDOFF, 0.0)
        t = min(max(BLOCK_FRACTION * usable / max(dist,1e-6), 0.0), 1.0)
        bx = ex + t * dx
        by = ey + t * dy
        bx = min(max(bx, pursuer_radius), w - pursuer_radius)
        by = min(max(by, pursuer_radius), h - pursuer_radius)
        return bx, by

    def angle_between(v1x, v1y, v2x, v2y):
        a = math.atan2(v1y, v1x)
        b = math.atan2(v2y, v2x)
        return abs((a - b + math.pi) % (2 * math.pi) - math.pi)

    ###########################################################################
    # MULTI-PURSUER CBF FILTER
    ###########################################################################
    def cbf_filter_multi(ev_x, ev_y, v_des, pursuers):
        nonlocal cbf_is_active

        v = v_des.copy()
        intervened = False

        # special case overlap
        for pu in pursuers:
            if abs(ev_x - pu["x"]) < CBF_EPS and abs(ev_y - pu["y"]) < CBF_EPS:
                cbf_is_active = True
                return clamp_speed(1.0, 0.0, evader_speed)

        for _ in range(CBF_MAX_ITERS):

            worst_margin = float('inf')
            worst_grad = None
            worst_h = None
            worst_vp = None

            for pu in pursuers:

                rx = ev_x - pu["x"]
                ry = ev_y - pu["y"]

                h = rx*rx + ry*ry - CBF_SAFETY_RADIUS**2
                grad_h = np.array([2.0*rx, 2.0*ry])
                v_p = np.array([pu["vx"], pu["vy"]])
                margin = float(grad_h @ (v - v_p) + CBF_ALPHA * h)

                if margin < worst_margin:
                    worst_margin = margin
                    worst_grad = grad_h
                    worst_h = h
                    worst_vp = v_p

            if worst_margin >= 0:
                break

            denom = float(worst_grad @ worst_grad)
            if denom < CBF_EPS:
                continue
            hdot = float(worst_grad @ (v - worst_vp))
            lam = (-(hdot + CBF_ALPHA * worst_h)) / denom
            v = v + lam * worst_grad
            intervened = True

        cbf_is_active = intervened
        return clamp_speed(v[0], v[1], evader_speed)

    ###########################################################################
    # START SIMULATION
    ###########################################################################

    print("\n=== Running Black Swan Simulation ===")

    while running and not game_over:

        current_time = pygame.time.get_ticks() / 1000.0 - start_time

        # quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Distances to pursuers
        d1 = math.hypot(evader_x - p1_x, evader_y - p1_y)
        d2 = math.hypot(evader_x - p2_x, evader_y - p2_y) if p2_active else float('inf')
        nearest = min(d1, d2)
        result.distance_trace.append(nearest)

        # Capture check
        if nearest <= COLLISION_DISTANCE:
            result.caught = True
            result.time_elapsed = current_time
            game_over = True
            print(f"Evader caught at {current_time:.2f}s!")
            break

        ###################################################################
        # Evader desired velocity
        ###################################################################
        gdx = goal_x - evader_x
        gdy = goal_y - evader_y
        gdist = math.hypot(gdx, gdy)
        
        if gdist > 0:
            v_goal = np.array([gdx/gdist, gdy/gdist]) * evader_speed
        else:
            v_goal = np.zeros(2)

        evade_vec = np.zeros(2)
        dists = []

        if d1 > 0:
            evade_vec += np.array([(evader_x - p1_x)/d1, (evader_y - p1_y)/d1]) * (1.0/d1)
            dists.append(d1)

        if p2_active and d2 > 0:
            evade_vec += np.array([(evader_x - p2_x)/d2, (evader_y - p2_y)/d2]) * (1.0/d2)
            dists.append(d2)

        if np.linalg.norm(evade_vec) > 0:
            evade_vec = evade_vec / np.linalg.norm(evade_vec) * evader_speed

        nearest = min(dists) if dists else float('inf')
        w_goal, w_evade = blend_weights(nearest)

        v_des = w_goal * v_goal + w_evade * evade_vec
        mag = np.linalg.norm(v_des)
        if mag > 0:
            v_des = v_des / mag * evader_speed

        pursuers = [
            {"x": p1_x, "y": p1_y, "vx": p1_x - prev_p1_x, "vy": p1_y - prev_p1_y}
        ]
        if p2_active:
            pursuers.append(
                {"x": p2_x, "y": p2_y, "vx": p2_x - prev_p2_x, "vy": p2_y - prev_p2_y}
            )

        safe_vx, safe_vy = cbf_filter_multi(evader_x, evader_y, v_des, pursuers)

        ###################################################################
        # Update evader
        ###################################################################
        prev_evader_x, prev_evader_y = evader_x, evader_y
        evader_x += safe_vx
        evader_y += safe_vy
        evader_x = max(evader_radius, min(w - evader_radius, evader_x))
        evader_y = max(evader_radius, min(h - evader_radius, evader_y))

        ###################################################################
        # Update Pursuer 1
        ###################################################################
        prev_p1_x, prev_p1_y = p1_x, p1_y
        dx1 = evader_x - p1_x
        dy1 = evader_y - p1_y
        d1 = math.hypot(dx1, dy1)
        if d1 > 0:
            p1_x += (dx1 / d1) * p1_speed
            p1_y += (dy1 / d1) * p1_speed
        p1_x = max(pursuer_radius, min(w - pursuer_radius, p1_x))
        p1_y = max(pursuer_radius, min(h - pursuer_radius, p1_y))

        ###################################################################
        # Update Pursuer 2 (blocker)
        ###################################################################
        if p2_active:
            prev_p2_x, prev_p2_y = p2_x, p2_y

            bx, by = goal_block_point((evader_x, evader_y), (goal_x, goal_y))

            v1x, v1y = p1_x - evader_x, p1_y - evader_y
            v2x, v2y = bx - evader_x, by - evader_y
            ang = angle_between(v1x, v1y, v2x, v2y)

            if ang < ANGLE_SEP_MIN:
                sgn = -1.0 if (v1x * (-v2y) + v1y * v2x) > 0 else 1.0
                rot90x = -v2y * sgn
                rot90y =  v2x * sgn
                m = math.hypot(rot90x, rot90y)
                if m > 1e-6:
                    rot90x /= m
                    rot90y /= m
                    offset = 60
                    bx += rot90x * offset
                    by += rot90y * offset

            dx2 = bx - p2_x
            dy2 = by - p2_y
            d2 = math.hypot(dx2, dy2)
            if d2 > 0:
                p2_x += (dx2 / d2) * p2_speed * BLOCK_REPOSITION_GAIN
                p2_y += (dy2 / d2) * p2_speed * BLOCK_REPOSITION_GAIN

        ###################################################################
        # Capture checks
        ###################################################################
        if math.hypot(evader_x - p1_x, evader_y - p1_y) <= COLLISION_DISTANCE:
            result.caught = True
            result.time_elapsed = current_time
            game_over = True
        
        if p2_active and math.hypot(evader_x - p2_x, evader_y - p2_y) <= COLLISION_DISTANCE:
            result.caught = True
            result.time_elapsed = current_time
            game_over = True

        ###################################################################
        # Goal collection + BLACK SWAN trigger
        ###################################################################
        gdist = math.hypot(evader_x - goal_x, evader_y - goal_y)
        if not game_over and goal_counter < max_goals and gdist <= evader_radius + goal_radius:

            result.goal_times.append(current_time)
            goal_counter += 1
            print(f"Goal {goal_counter} reached at {current_time:.2f}s")

            # ---- BLACK SWAN TRIGGER ----
            if (not p2_active) and (goal_counter == black_swan_goal_index):
                p2_active = True
                result.black_swan_trigger_goal = goal_counter  # <---- NEW METRIC LOGGING

                p2_x, p2_y = spawn_far_from_evader(min_dist=CBF_SAFETY_RADIUS * 1.5)
                prev_p2_x, prev_p2_y = p2_x, p2_y

                black_swan_announced = True
                announce_timer = 120

                p1_speed = PURSUER_SPEED_COOP
                p2_speed = PURSUER_SPEED_COOP

                print(f"⚠ BLACK SWAN: Pursuer 2 spawned at GOAL {goal_counter}")

            # Goal completed or continue
            if goal_counter >= max_goals:
                result.goals_reached = goal_counter
                result.time_elapsed = current_time
                game_over = True
            else:
                goal_x, goal_y = random.randint(0, w), random.randint(0, h)

        ###################################################################
        # Time-out
        ###################################################################
        if current_time >= max_time:
            result.time_elapsed = current_time
            result.goals_reached = goal_counter
            game_over = True
            print(f"Time limit reached at {max_time}s")
            break

        ###################################################################
        # RENDERING
        ###################################################################
        screen.fill(WHITE)

        # Draw safety + stray zones
        pygame.draw.circle(screen, LIGHT_BLUE, (int(p1_x), int(p1_y)), int(CBF_SAFETY_RADIUS), 2)
        pygame.draw.circle(screen, LIGHT_GRAY, (int(p1_x), int(p1_y)), int(STRAY_RADIUS), 1)

        if p2_active:
            pygame.draw.circle(screen, LIGHT_BLUE, (int(p2_x), int(p2_y)), int(CBF_SAFETY_RADIUS), 2)
            pygame.draw.circle(screen, LIGHT_GRAY, (int(p2_x), int(p2_y)), int(STRAY_RADIUS), 1)

            bx, by = goal_block_point((evader_x, evader_y), (goal_x, goal_y))
            pygame.draw.circle(screen, TEAL, (int(bx), int(by)), 5)
            pygame.draw.line(screen, BLACK, (int(p2_x), int(p2_y)), (int(bx), int(by)), 1)

        pygame.draw.circle(screen, RED, (int(p1_x), int(p1_y)), pursuer_radius)
        if p2_active:
            pygame.draw.circle(screen, ORANGE, (int(p2_x), int(p2_y)), pursuer_radius)
        pygame.draw.circle(screen, BLUE, (int(evader_x), int(evader_y)), evader_radius)

        pygame.draw.line(screen, BLACK, (int(evader_x), int(evader_y)), (int(p1_x), int(p1_y)), 1)

        if goal_counter < max_goals:
            pygame.draw.circle(screen, GREEN, (int(goal_x), int(goal_y)), goal_radius)

        # HUD
        screen.blit(font.render(f"Goals: {goal_counter}", True, GREEN), (10, 10))
        screen.blit(font.render(f"Time: {current_time:.1f}s", True, BLACK), (10, 50))

        if p2_active:
            screen.blit(font.render("Cooperative Pursuit Active", True, PURPLE), (10, 90))

        if cbf_is_active:
            screen.blit(font.render("CBF ACTIVE", True, RED), (10, 130))

        if black_swan_announced and announce_timer > 0:
            msg = font.render("⚠ BLACK SWAN!", True, ORANGE)
            screen.blit(msg, msg.get_rect(center=(w//2, 40)))
            announce_timer -= 1
            if announce_timer == 0:
                black_swan_announced = False

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return result.to_dict()

###############################################################################
# MULTIPLE TRIAL WRAPPER
###############################################################################

def run_multiple_black_swan_trials(num_trials: int = 15, max_goals: int = 10, max_time: float = 30.0):

    print("\nRunning Black Swan Trials...")
    print("==================================================")

    all_results = []

    for trial in range(num_trials):
        print(f"\n=== Trial {trial + 1}/{num_trials} ===")
        result = run_black_swan_filter(
            seed=trial,
            max_goals=max_goals,
            max_time=max_time
        )

        all_results.append(result)

        print(f"  Goals reached: {result['goals_reached']}")
        print(f"  Status: {'Caught' if result['caught'] else 'Completed'}")
        print(f"  Time: {result['time_elapsed']:.2f}s")
        print(f"  Black Swan triggered at goal: {result['black_swan_trigger_goal']}")

    filename = f"black_swan_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filename, "w") as f:
        json.dump({"trials": all_results}, f, indent=2)

    print(f"\nResults saved to {filename}")

    return all_results

###############################################################################
# MAIN
###############################################################################

if __name__ == "__main__":
    run_multiple_black_swan_trials(num_trials=15)

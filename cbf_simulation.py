import pygame
import math
import random
import numpy as np
import json
import datetime
from typing import Dict, Any, List

class SimulationResult:
    def __init__(self):
        self.goals_reached: int = 0
        self.caught: bool = False
        self.time_elapsed: float = 0.0
        self.distance_trace: List[float] = []
        self.goal_times: List[float] = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'goals_reached': self.goals_reached,
            'caught': self.caught,
            'time_elapsed': self.time_elapsed,
            'distance_trace': self.distance_trace,
            'goal_times': self.goal_times
        }

#######################################################################
# RUN CBF SIMULATION (MATCHES SWITCHING FILTER VERSION STRUCTURE)
#######################################################################

def run_cbf_filter(seed: int = None, max_goals: int = 10, max_time: float = 30.0) -> Dict[str, Any]:

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    result = SimulationResult()

    ########################################################
    # Settings (same as your standalone CBF file)
    ########################################################
    w, h = 800, 600
    evader_speed = 8
    pursuer_speed = 1
    evader_radius = pursuer_radius = 15
    goal_radius = 20
    COLLISION_DISTANCE = evader_radius + pursuer_radius

    # CBF parameters
    CBF_SAFETY_RADIUS = 80
    CBF_ALPHA = 0.5
    STRAY_RADIUS = 150

    # Colors
    WHITE, BLUE, RED, GREEN, BLACK = (255,255,255), (50,150,255), (255,80,80), (50,200,50), (0,0,0)
    PURPLE = (150, 0, 150)

    ########################################################
    # Pygame initialization
    ########################################################
    pygame.init()
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("CBF Filter Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    ########################################################
    # Initial positions
    ########################################################
    evader_x = w // 4
    evader_y = h // 2
    pursuer_x = 3 * w // 4
    pursuer_y = h // 2

    # Goal initialization
    goal_x, goal_y = random.randint(0, w), random.randint(0, h)
    goal_spawn_time = pygame.time.get_ticks() / 1000.0
    goal_counter = 0

    running, game_over = True, False
    cbf_is_active = False
    w_goal, w_evade = 1.0, 0.0

    start_time = pygame.time.get_ticks() / 1000.0

    ########################################################
    # CBF FILTER
    ########################################################
    def cbf_filter(ev_x, ev_y, p_x, p_y, des_vx, des_vy):
        nonlocal cbf_is_active

        rel_x = ev_x - p_x
        rel_y = ev_y - p_y
        h = rel_x**2 + rel_y**2 - CBF_SAFETY_RADIUS**2
        h_dot = 2 * (rel_x * des_vx + rel_y * des_vy)

        if h_dot >= -CBF_ALPHA * h:
            cbf_is_active = False
            return des_vx, des_vy

        cbf_is_active = True
        dist = np.sqrt(rel_x**2 + rel_y**2)
        if dist == 0:
            return 0, 0

        norm_x = rel_x / dist
        norm_y = rel_y / dist
        v_normal_scalar = des_vx * norm_x + des_vy * norm_y

        v_tangent_x = des_vx - v_normal_scalar * norm_x
        v_tangent_y = des_vy - v_normal_scalar * norm_y

        return v_tangent_x, v_tangent_y

    ########################################################
    # MAIN SIMULATION LOOP
    ########################################################
    print("\n=== Running CBF Simulation ===")

    while running and not game_over:

        current_time = pygame.time.get_ticks() / 1000.0 - start_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Compute distance
        dist_to_pursuer = math.hypot(evader_x - pursuer_x, evader_y - pursuer_y)
        result.distance_trace.append(dist_to_pursuer)

        # Capture check
        if dist_to_pursuer <= COLLISION_DISTANCE:
            result.caught = True
            result.time_elapsed = current_time
            game_over = True
            print(f"Evader caught at {current_time:.2f}s!")
            break

        ####################################################
        # Compute desired velocities
        ####################################################
        goal_dx = goal_x - evader_x
        goal_dy = goal_y - evader_y
        goal_dist = math.hypot(goal_dx, goal_dy)

        if goal_dist > 0:
            v_goal_x = (goal_dx / goal_dist) * evader_speed
            v_goal_y = (goal_dy / goal_dist) * evader_speed
        else:
            v_goal_x = v_goal_y = 0

        evade_dx = evader_x - pursuer_x
        evade_dy = evader_y - pursuer_y
        evade_dist = math.hypot(evade_dx, evade_dy)

        if evade_dist > 0:
            v_evade_x = (evade_dx / evade_dist) * evader_speed
            v_evade_y = (evade_dy / evade_dist) * evader_speed
        else:
            v_evade_x = v_evade_y = 0

        # Weighting logic
        if dist_to_pursuer > STRAY_RADIUS:
            w_goal, w_evade = 1.0, 0.0
        elif dist_to_pursuer < CBF_SAFETY_RADIUS:
            w_goal, w_evade = 0.0, 1.0
        else:
            w_evade = 1.0 - (dist_to_pursuer - CBF_SAFETY_RADIUS) / (STRAY_RADIUS - CBF_SAFETY_RADIUS)
            w_goal = 1.0 - w_evade

        desired_vx = v_goal_x * w_goal + v_evade_x * w_evade
        desired_vy = v_goal_y * w_goal + v_evade_y * w_evade

        mag = math.hypot(desired_vx, desired_vy)
        if mag > 0:
            desired_vx = desired_vx / mag * evader_speed
            desired_vy = desired_vy / mag * evader_speed

        safe_vx, safe_vy = cbf_filter(evader_x, evader_y, pursuer_x, pursuer_y, desired_vx, desired_vy)

        ####################################################
        # Update positions
        ####################################################
        evader_x += safe_vx
        evader_y += safe_vy
        evader_x = max(evader_radius, min(w - evader_radius, evader_x))
        evader_y = max(evader_radius, min(h - evader_radius, evader_y))

        dx = evader_x - pursuer_x
        dy = evader_y - pursuer_y
        d = math.hypot(dx, dy)
        if d > 0:
            pursuer_x += (dx / d) * pursuer_speed
            pursuer_y += (dy / d) * pursuer_speed

        pursuer_x = max(pursuer_radius, min(w - pursuer_radius, pursuer_x))
        pursuer_y = max(pursuer_radius, min(h - pursuer_radius, pursuer_y))

        ####################################################
        # Goal check
        ####################################################
        gdist = math.hypot(evader_x - goal_x, evader_y - goal_y)
        if goal_counter < max_goals and gdist <= evader_radius + goal_radius:
            result.goal_times.append(current_time - (goal_spawn_time - start_time))
            goal_counter += 1
            print(f"Goal {goal_counter} reached at {current_time:.2f}s")

            if goal_counter >= max_goals:
                result.goals_reached = goal_counter
                result.time_elapsed = current_time
                game_over = True
                print("All goals collected!")
            else:
                goal_x, goal_y = random.randint(0, w), random.randint(0, h)
                goal_spawn_time = pygame.time.get_ticks() / 1000.0

        ####################################################
        # Time-out check
        ####################################################
        if current_time >= max_time:
            result.goals_reached = goal_counter
            result.time_elapsed = current_time
            game_over = True
            print(f"Time's up at {max_time}s.")
            break

        ####################################################
        # Rendering
        ####################################################
        screen.fill(WHITE)
        pygame.draw.circle(screen, BLUE, (int(evader_x), int(evader_y)), evader_radius)
        pygame.draw.circle(screen, RED, (int(pursuer_x), int(pursuer_y)), pursuer_radius)
        pygame.draw.line(screen, BLACK, (int(evader_x), int(evader_y)), (int(pursuer_x), int(pursuer_y)), 1)

        if goal_counter < max_goals:
            pygame.draw.circle(screen, GREEN, (int(goal_x), int(goal_y)), goal_radius)

        pygame.draw.circle(screen, (200,200,255), (int(pursuer_x), int(pursuer_y)), CBF_SAFETY_RADIUS, 2)
        pygame.draw.circle(screen, (230,230,230), (int(pursuer_x), int(pursuer_y)), STRAY_RADIUS, 1)

        screen.blit(font.render(f"Goals: {goal_counter}/{max_goals}", True, GREEN), (10, 10))
        screen.blit(font.render("Filter: CBF", True, PURPLE), (10, 50))
        screen.blit(font.render(f"Time: {current_time:.1f}s", True, BLACK), (10, 90))

        if cbf_is_active:
            screen.blit(font.render("CBF ACTIVE", True, RED), (10, 130))
        else:
            screen.blit(font.render("SLIDING / NORMAL", True, BLACK), (10, 130))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return result.to_dict()


#######################################################################
# MULTIPLE TRIAL WRAPPER (MATCHES SWITCHING VERSION)
#######################################################################

def run_multiple_cbf_trials(num_trials: int = 15, max_goals: int = 10, max_time: float = 30.0):
    print("\nRunning CBF Filter Trials...")
    print("==================================================")

    all_results = []

    for trial in range(num_trials):
        seed = trial
        print(f"\n=== Trial {trial+1}/{num_trials} ===")

        result = run_cbf_filter(
            seed=seed,
            max_goals=max_goals,
            max_time=max_time
        )

        all_results.append(result)

        print(f"  Goals: {result['goals_reached']}/{max_goals}")
        print(f"  Status: {'Caught' if result['caught'] else 'Completed'}")
        print(f"  Time: {result['time_elapsed']:.2f}s")

    # Save results
    filename = f"cbf_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump({"trials": all_results}, f, indent=2)

    print(f"\nResults saved to {filename}")
    return all_results


#######################################################################
# MAIN
#######################################################################

if __name__ == "__main__":
    run_multiple_cbf_trials(num_trials=15)

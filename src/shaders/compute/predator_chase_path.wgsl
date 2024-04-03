const NUM_BOIDS: u32 = 255u;
const NUM_PREDATORS: u32 = 4u;

const MAX_SCREEN_X: f32 = 150.0;
const MIN_SCREEN_X: f32 = -150.0;
const MAX_SCREEN_Y: f32 = 70.0;
const MIN_SCREEN_Y: f32 = -70.0;

struct PreyData {
  id: u32,
  dist: f32,
}
struct Boid {
  pos: vec2<f32>,
  vel: vec2<f32>,
}
struct PredatorParams {
  max_velocity: f32,
  min_velocity: f32,
  turn_factor: f32,
  pursuit_factor: f32,
  pursuit_multiplier: f32,
  matching_factor: f32,
  self_avoid_factor: f32,
  visual_range: f32,
  protected_range: f32,
  interest_range: f32,
}

@group(0) @binding(0) var<storage, read_write> boids: array<Boid>;
@group(0) @binding(2) var<storage, read_write> predators: array<Boid>;
@group(0) @binding(3) var<storage, read_write> pp: PredatorParams;
@group(0) @binding(5) var<storage, read_write> pursuits: array<u32>;
@group(0) @binding(6) var<storage, read_write> captures: array<u32>;
@group(0) @binding(7) var<storage, read_write> captured: array<f32>;

fn seperation(predator: Boid) -> vec2<f32> {
  var ip = predator;
  var close_dx = 0.0;
  var close_dy = 0.0;
  var dv = vec2(0.0);

  for (var i: u32 = 0u; i < NUM_PREDATORS; i++) {
    let neighbour_dist: f32 = distance(ip.pos, predators[i].pos);

    if (neighbour_dist < pp.protected_range) {
      close_dx += ip.pos.x - predators[i].pos.x;
      close_dy += ip.pos.y - predators[i].pos.y;
    }
  }

  dv.x += close_dx * pp.self_avoid_factor;
  dv.y += close_dy * pp.self_avoid_factor;

  return dv;
}

fn find_closest_boid(predator: Boid) -> u32 {
  var closest_id: u32 = 0xFFFFFFFFu;
  var closest_dist: f32 = 99999999.0;

  for (var i: u32 = 0u; i < NUM_BOIDS; i++) {
    let dist = distance(predator.pos, boids[i].pos);
    
    if (dist < closest_dist*captured[i]) {
      closest_dist = dist;
      closest_id = i;
    }
  }

  return closest_id;
}

fn find_flock_center() -> vec2<f32> {
  var avg_pos = vec2(0.0);
  var num_captured = f32(NUM_BOIDS);

  for (var i: u32 = 0u; i < NUM_BOIDS; i++) {
    avg_pos += boids[i].pos*captured[i];
    num_captured -= captured[i];
  }
  avg_pos /= (f32(NUM_BOIDS) - num_captured);

  return avg_pos;
}

fn find_lead_boid() -> u32 {
  var avg_vel = vec2(0.0);
  var num_captured = f32(NUM_BOIDS);

  for (var i: u32 = 0u; i < NUM_BOIDS; i++) {
    avg_vel += boids[i].vel*captured[i];
    num_captured -= captured[i];
  }

  avg_vel /= (f32(NUM_BOIDS) - num_captured);
  
  avg_vel = normalize(avg_vel);

  var max_dp = -1.0;
  var lead_id = 0u;

  for (var i: u32; i < NUM_BOIDS; i++) {
    let dp = dot(boids[i].vel, avg_vel)*captured[i];

    if (dp > max_dp) {
      max_dp = dp;
      lead_id = i;
    }
  }

  return lead_id;
}

fn predict_lead_pos(boid: Boid) -> vec2<f32> {
  let predict_step_size = 5.0;
  return boid.pos + (predict_step_size*boid.vel);
}

fn find_outermost_boid() -> u32 {
  let fc = find_flock_center();
  var max_dist = 0.0;
  var ffc_id = 0u;

  for (var i: u32 = 0u; i < NUM_BOIDS; i++) {
    let dist_from_center = distance(fc, boids[i].pos);
    if (dist_from_center*captured[i] > max_dist) {
      max_dist = dist_from_center;
      ffc_id = i;
    }
  }

  return ffc_id;
}

fn get_prey_direction(prey_pos: vec2<f32>, predator: Boid) -> vec2<f32> {
  let dv_norm = normalize(prey_pos - predator.pos);

  return dv_norm * pp.pursuit_factor;
}

fn match_velocity(prey_vel: vec2<f32>, predator: Boid) -> vec2<f32> {
  return prey_vel*pp.matching_factor;
}

fn respect_screen_edges(predator: Boid) -> vec2<f32> {
  var dv = vec2(0.0);

  if (predator.pos.x < MIN_SCREEN_X) {
    dv.x += pp.turn_factor;
  }
  if (predator.pos.x > MAX_SCREEN_X) {
    dv.x -= pp.turn_factor;
  }
  if (predator.pos.y < MIN_SCREEN_Y) {
    dv.y += pp.turn_factor;
  }
  if (predator.pos.y > MAX_SCREEN_Y) {
    dv.y -= pp.turn_factor;
  }

  return dv;
}

fn respect_speed_limit(predator: Boid) -> vec2<f32> {
  return clamp(predator.vel, vec2(pp.min_velocity), vec2(pp.max_velocity));
}

fn check_captures(pid: u32, predator: Boid) {
  for (var i: u32 = 0u; i < NUM_BOIDS; i++) {
    let dist = distance(predator.pos, boids[i].pos);

    if (dist < 5.0 && captured[i] != 0.0) {
      captures[pid]++;
      captured[i] = 0.0;
      break;
    }
  }
}

@compute 
@workgroup_size(4, 1, 1) 
fn compute_predator_pursuit(@builtin(global_invocation_id) id: vec3<u32>) {
  // If already in pursuit continue;
  if (pursuits[id.x] != 0xFFFFFFFFu) {
    let chasing_id = pursuits[id.x];
    // if boid beyond interest range then stop pursuing and employ other strategies
    if (distance(boids[chasing_id].pos, predators[id.x].pos) > pp.interest_range) {
      pursuits[id.x] = 0xFFFFFFFFu;
    } else {
      predators[id.x].vel += get_prey_direction(boids[chasing_id].pos, predators[id.x])*pp.pursuit_multiplier;
      predators[id.x].vel += match_velocity(boids[chasing_id].vel, predators[id.x])*pp.pursuit_multiplier;
    }
  } else {
    let closest_id: u32 = find_closest_boid(predators[id.x]);
    let closest_boid = boids[closest_id];
    // Predator 0 targets center of mass of flock 
    if (id.x == 0u) {
      let fc = find_flock_center();
      predators[id.x].vel += get_prey_direction(fc, predators[id.x]);

    // Predator 1 targets the closest boid to itself 
    } else if (id.x == 1u) {
      predators[id.x].vel += get_prey_direction(closest_boid.pos, predators[id.x]);
      predators[id.x].vel += match_velocity(closest_boid.vel, predators[id.x]);

    // Predator 2 targets the lead boid
    } else if (id.x == 2u) {
      let lead_id: u32 = find_lead_boid();
      let lead_boid = boids[lead_id];
      predators[id.x].vel += get_prey_direction(lead_boid.pos, predators[id.x]);
      predators[id.x].vel += match_velocity(lead_boid.vel, predators[id.x]);

    // Predator 3 targets the boid that is furthest from the center
    } else {
      let outermost_id: u32 = find_outermost_boid();
      let outermost_boid = boids[outermost_id]; 
      predators[id.x].vel += get_prey_direction(outermost_boid.pos, predators[id.x]);
      predators[id.x].vel += match_velocity(outermost_boid.vel, predators[id.x]);
    }

    // All Predators - If a boid comes within interest range pursue it in next cycle
    if (distance(boids[closest_id].pos, predators[id.x].pos) < pp.interest_range) {
      pursuits[id.x] = closest_id;
    }
  }

  // Dont bump into each other, exceed screen limits or speed limits
  predators[id.x].vel += seperation(predators[id.x]);
  predators[id.x].vel += respect_screen_edges(predators[id.x]);
  predators[id.x].vel = respect_speed_limit(predators[id.x]);

  predators[id.x].pos += predators[id.x].vel;
  
  // Captures the first boid within 5.0 that shows up in the loop
  check_captures(id.x, predators[id.x]);
}

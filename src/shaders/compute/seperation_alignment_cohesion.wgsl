const NUM_BOIDS: u32 = 255u;
const NUM_PREDATORS: u32 = 4u;

const MAX_SCREEN_X: f32 = 150.0;
const MIN_SCREEN_X: f32 = -150.0;
const MAX_SCREEN_Y: f32 = 70.0;
const MIN_SCREEN_Y: f32 = -70.0;

const MAX_BIAS: f32 = 0.01;
const BIAS_VAL: f32 = 0.001;
const BIAS_INCREMENT: f32 = 0.00004;

struct Boid {
  pos: vec2<f32>,
  vel: vec2<f32>,
}
struct BoidParams {
  max_velocity: f32,
  min_velocity: f32,
  turn_factor: f32,
  visual_range: f32,
  protected_range: f32,
  centering_factor: f32,
  self_avoid_factor: f32,
  predator_avoid_factor: f32,
  matching_factor: f32,
}
struct TimeUniform {
  time: f32,
}

@group(0) @binding(0) var<storage, read_write> boids: array<Boid>;
@group(0) @binding(1) var<storage, read_write> bp: BoidParams;
@group(0) @binding(2) var<storage, read_write> predators: array<Boid>;
@group(0) @binding(4) var<uniform> tu: TimeUniform;
@group(0) @binding(7) var<storage, read_write> captured: array<f32>;

fn seperation(boid: Boid) -> vec2<f32> {
  var ib = boid;
  var close_dx = 0.0;
  var close_dy = 0.0;
  var dv = vec2(0.0);

  for (var i: u32 = 0u; i < NUM_BOIDS; i++) {
    let neighbour_dist: f32 = distance(ib.pos, boids[i].pos);

    if (neighbour_dist < bp.protected_range) {
      close_dx += (ib.pos.x - boids[i].pos.x)*captured[i];
      close_dy += (ib.pos.y - boids[i].pos.y)*captured[i];
    }
  }

  dv.x += close_dx * bp.self_avoid_factor;
  dv.y += close_dy * bp.self_avoid_factor;

  return dv;
}

fn avoid_predators(boid: Boid) -> vec2<f32> {
  var dv = vec2(0.0);
  
  for (var i: u32 = 0u; i < NUM_PREDATORS; i++) {
    if (distance(boid.pos, predators[i].pos) < bp.visual_range) {
      dv.x += (boid.pos.x - predators[i].pos.x)*bp.predator_avoid_factor;
      dv.y += (boid.pos.y - predators[i].pos.y)*bp.predator_avoid_factor;
    }
  }

  return dv;
}

fn alignment(boid: Boid) -> vec2<f32> {
  var vx_avg = 0.0;
  var vy_avg = 0.0;
  var num_neighbours = 0.0;

  for (var i: u32 = 0u; i < NUM_BOIDS; i++) {
    if (distance(boids[i].pos, boid.pos) < bp.visual_range) {
      vx_avg += boids[i].vel.x*captured[i];
      vy_avg += boids[i].vel.y*captured[i];
      num_neighbours += 1.0*captured[i];
    }
  }

  vx_avg = vx_avg / num_neighbours;
  vy_avg = vy_avg / num_neighbours;

  let dvx = (vx_avg - boid.vel.x)*bp.matching_factor;
  let dvy = (vy_avg - boid.vel.y)*bp.matching_factor;

  return vec2<f32>(dvx, dvy);
}

fn cohesion(boid: Boid) -> vec2<f32> {
  var x_avg = 0.0;
  var y_avg = 0.0;
  var num_neighbours = 0.0;

  for (var i: u32 = 0u; i < NUM_BOIDS; i++) {
    x_avg += boids[i].pos.x*captured[i];
    y_avg += boids[i].pos.y*captured[i];
    num_neighbours += 1.0*captured[i];
  }

  x_avg = x_avg / num_neighbours;
  y_avg = y_avg / num_neighbours;

  let dx = (x_avg - boid.pos.x)*bp.centering_factor;
  let dy = (y_avg - boid.pos.y)*bp.centering_factor;

  return vec2<f32>(dx, dy);
}

fn respect_screen_edges(boid: Boid) -> vec2<f32> {
  var dv = vec2(0.0);

  if (boid.pos.x < MIN_SCREEN_X) {
    dv.x += bp.turn_factor;
  }
  if (boid.pos.x > MAX_SCREEN_X) {
    dv.x -= bp.turn_factor;
  }
  if (boid.pos.y < MIN_SCREEN_Y) {
    dv.y += bp.turn_factor;
  }
  if (boid.pos.y > MAX_SCREEN_Y) {
    dv.y -= bp.turn_factor;
  }

  return dv;
}

fn respect_speed_limit(boid: Boid) -> vec2<f32> {
  return clamp(boid.vel, vec2(bp.min_velocity), vec2(bp.max_velocity));
}

@compute 
@workgroup_size(16, 16, 1) 
fn sac(@builtin(global_invocation_id) id: vec3<u32>) {
  boids[id.x].vel += seperation(boids[id.x]);
  boids[id.x].vel += avoid_predators(boids[id.x]);
  boids[id.x].vel += alignment(boids[id.x]);
  boids[id.x].vel += cohesion(boids[id.x]);

  boids[id.x].vel += respect_screen_edges(boids[id.x]);
  boids[id.x].vel = respect_speed_limit(boids[id.x]);
  
  boids[id.x].pos += boids[id.x].vel;
}

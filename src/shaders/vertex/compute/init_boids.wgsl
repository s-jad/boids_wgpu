struct Boid {
  pos: vec2<f32>,
  vel: vec2<f32>,
}
@group(0) @binding(0) var<storage, read_write> boids: array<Boid>;

struct RandomResult {
    state: vec4<u32>,
    value: f32,
};

fn taus_step(z: u32, S1: u32, S2: u32, S3: u32, M: u32) -> u32 {
    let b = (((z << S1) ^ z) >> S2);
    return ((z & M) << S3) ^ b;
}

fn lcg_step(z: u32, A: u32, C: u32) -> u32 {
    return A * z + C;
}

fn hybrid_taus(st: vec4<u32>) -> RandomResult {
    var state = st; 
    state.x = taus_step(state.x, 13u, 19u, 12u, 4294967294u);
    state.y = taus_step(state.y, 2u, 25u, 4u, 4294967288u);
    state.z = taus_step(state.z, 3u, 11u, 17u, 4294967280u);
    state.w = lcg_step(state.w, 1664525u, 1013904223u);

    var rand: RandomResult;
    rand.state = state;
    rand.value = 2.0  * f32(state.x ^ state.y ^ state.z ^ state.w) / f32(0xFFFFFFFFu) - 1.0;

    return rand;
}

@compute 
@workgroup_size(16, 16, 1) 
fn compute_boid_positions(@builtin(global_invocation_id) id: vec3<u32>) {
  let seed = id.x * 1000u + id.y * 100u + id.z;
  let state = vec4<u32>(seed, seed + 1u, seed + 2u, seed + 3u);

  // Random pos(x,y)
  var rx: RandomResult = hybrid_taus(state);
  var ry: RandomResult = hybrid_taus(rx.state);
  // Random vel(x,y)
  var rvx: RandomResult = hybrid_taus(ry.state);
  var rvy: RandomResult = hybrid_taus(rvx.state);

  boids[id.x].pos = vec2<f32>(rx.value * 100.0, ry.value * 70.0);
  boids[id.x].vel = vec2<f32>(rvx.value, rvy.value) * 0.2;
}

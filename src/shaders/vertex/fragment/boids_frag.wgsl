// CONSTANTS
const PI: f32 = 3.14159265;
const NUM_BOIDS: u32 = 255u;
const NUM_PREDATORS: u32 = 4u;

// STRUCTS
struct TimeUniform {
    time: f32,
};
struct ResolutionUniform {
  xy: vec2<f32>,
}
struct ViewParameters {
  x_shift: f32,
  y_shift: f32,
  zoom: f32,
  time_modifier: f32,
}
struct Boid {
  pos: vec2<f32>,
  vel: vec2<f32>,
}

// GROUPS AND BINDINGS
@group(0) @binding(0)
var<storage, read> boids: array<Boid>;
@group(0) @binding(2)
var<storage, read> predators: array<Boid>;
@group(0) @binding(6)
var<storage, read> captures: array<u32>;
@group(0) @binding(7)
var<storage, read> captured: array<f32>;

@group(1) @binding(0)
var<uniform> tu: TimeUniform;
@group(1) @binding(1)
var<uniform> ru: ResolutionUniform;

@group(2) @binding(0)
var<storage, read_write> pa: ViewParameters;

// ASPECT RATIO
fn scale_aspect(fc: vec2<f32>) -> vec2<f32> {
  // Scale from 0.0 --> 1.0 to -1.0 --> 1.0 
  var uv: vec2<f32> = ((fc * 2.0) - screen) / max(screen.x, screen.y);
  uv.y *= -1.0;
  return uv;
}

// COLORS
fn palette(t: f32) -> vec3<f32> {
  let a: vec3<f32> = vec3<f32>(0.120, 0.618, 0.624); 
  let b: vec3<f32> = vec3<f32>(0.878, 0.214, 0.229);
  let c: vec3<f32> = vec3<f32>(0.654, 0.772, 0.426);
  let d: vec3<f32> = vec3<f32>(0.937, 0.190, 0.152);

  return a * b * cos(PI * 2.0 * (c * t + d));
}

// HASHING
fn shash21(pos: vec2<f32>) -> f32 {
  return fract(sin(dot(pos, vec2(12.34777, 67.8913375))) * 4277123.455) * 2.0 - 1.0;
}

fn shash22(pos: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(
    fract(sin(dot(pos, vec2(12.34777, 67.8913375))) * 4277123.455),
    fract(cos(dot(pos, vec2(71.43119, 33.7654637))) * 9854756.117)
  ) * 2.0 - 1.0;
}


fn rotate(v: vec2<f32>, a: f32, tc: vec2<f32>) -> vec2<f32> {
    let ca = cos(a);
    let sa = sin(a);
    let dv = v - tc;
    let M = mat2x2(ca, -sa, sa, ca);

    let rotated = dv*M;
    return rotated;
}

fn barycentric_triangle(uv: vec2<f32>, boid: Boid, clr: u32, p_size: u32) -> vec3<f32> {
    let tc = boid.pos; // Center of the triangle
    let a = 1.5+(0.07*f32(1u + p_size)); // Half-width of the triangle
    let b = 3.0+(0.07*f32(1u + p_size)); // Half-height of the triangle

    let v0 = vec2<f32>(tc.x, tc.y);
    let v1 = vec2<f32>(tc.x + a, tc.y + b);
    let v2 = vec2<f32>(tc.x - a, tc.y + b);

    // Calculate barycentric coordinates
    let p = uv;
    var rv0 = v1 - v0;
    var rv1 = v2 - v0;
    var rv2 = p - v0;

    //// Calculate the angle of the velocity vector
    //let angle = atan2(boid.vel.x, boid.vel.y);

    //// Rotate the triangle vertices based on the angle
    //rv0 = rotate(rv0, angle, tc);
    //rv1 = rotate(rv1, angle, tc);
    //rv2 = rotate(rv2, angle, tc);

    let d00 = dot(rv0, rv0);
    let d01 = dot(rv0, rv1);
    let d11 = dot(rv1, rv1);
    let d20 = dot(rv2, rv0);
    let d21 = dot(rv2, rv1);
    let inv_denom = 1.0 / (d00 * d11 - d01 * d01);
    let v = (d11 * d20 - d01 * d21) * inv_denom;
    let w = (d00 * d21 - d01 * d20) * inv_denom;
    let u = 1.0 - v - w;

    // Determine if the point is inside the triangle
    if (u >= 0.0) && (v >= 0.0) && (w >= 0.0) {
        return palette(f32(clr) + 12.0)*4.5; // Red color if inside the triangle
    } else {
        return vec3<f32>(0.0, 0.0, 0.0); // Black color if outside the triangle
    }
}

const screen: vec2<f32> = vec2(1366.4, 768.0);
@fragment
fn main(@builtin(position) FragCoord: vec4<f32>) -> @location(0) vec4<f32> {
  let t: f32 = tu.time * pa.time_modifier;
  let ts: f32 = sin(t);
  var uv: vec2<f32> = scale_aspect(FragCoord.xy); // Scale to -1.0 -> 1.0 + fix aspect ratio
  uv.x += pa.x_shift * pa.zoom;
  uv.y += pa.y_shift * pa.zoom;
  uv /= pa.zoom;
  var uv0 = uv;
  var color = vec3(0.0);
// -----------------------------------------------------------------------------------------------

  
  for (var i: u32 = 0u; i < NUM_BOIDS; i++) {
    let bd = distance(uv, boids[i].pos);
    // If boid caught, captured[i] == 0.0 and boid won't be visible
    color += captured[i] - smoothstep(0.0, 1.0, bd)*captured[i];
  }
  
  for (var i: u32 = 0u; i < NUM_PREDATORS; i++) {
    color += barycentric_triangle(uv, predators[i], i, captures[i]);
  }

// -----------------------------------------------------------------------------------------------

  
  return vec4<f32>(color, 1.0);
}
